import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import time
import logging
import pickle
from datetime import datetime
from pathlib import Path
import multiprocessing
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sentiment_pipeline")

warnings.filterwarnings("ignore")

# -------------------------------
# Configuration & File Paths
# -------------------------------
STOCK_DATA_PATH = r"A:\RIT\RIT 3rd Sem\Applied Data Science DSCI\Project\Stocks\SP500_with_Indicators.csv"
OUTPUT_CSV = "final_aggregated_sentiment_stock.csv"
STOCK_DATE_COLUMN = "Date"

# JSONL files with news comments
FOX_PATH = r"C:\Users\ankit\Downloads\fox-003.jsonl"
CNN_PATH = r"C:\Users\ankit\Downloads\cnn-001.jsonl"
MSNBC_PATH = r"C:\Users\ankit\Downloads\msnbc-002.jsonl"

# Processing configuration
MAX_COMMENTS_PER_DAY = 500  # Increased for better representation
CHECKPOINT_DIR = "checkpoints"
CHUNK_SIZE = 250000  # Increased chunk size for faster processing
BATCH_SIZE = 128  # Larger batch sizes for GPU efficiency
EARLY_EXIT_THRESHOLD = 0.85  # Reduced slightly to process more data

# Date range for filtering comments
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# GPU memory optimization
PRECISION = "fp16"  # Use mixed precision to reduce memory usage

# Create checkpoint directory if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------------
# Helper: Shift Weekends to Monday
# -------------------------------
def effective_date(dt):
    """Map weekend dates to next Monday for market alignment"""
    if dt.weekday() == 5:  # Saturday
        return dt + pd.Timedelta(days=2)
    elif dt.weekday() == 6:  # Sunday
        return dt + pd.Timedelta(days=1)
    else:
        return dt

# -------------------------------
# Count Chunks
# -------------------------------
def count_chunks(file_path, chunksize=CHUNK_SIZE):
    """Count the number of chunks in a JSONL file"""
    try:
        count = 0
        for _ in pd.read_json(file_path, lines=True, chunksize=chunksize):
            count += 1
        return count
    except Exception as e:
        logger.error(f"Error counting chunks in {file_path}: {e}")
        return 0

# -------------------------------
# Process File in Chunks
# -------------------------------
def process_file_chunks(file_path, max_comments=MAX_COMMENTS_PER_DAY, chunksize=CHUNK_SIZE, 
                        early_exit_threshold=EARLY_EXIT_THRESHOLD):
    """
    Reads JSONL in chunks, extracts up to max_comments per effective date (weekends->Monday),
    uses an early_exit_threshold to skip further lines if dates are "filled."
    Returns daily_comments: {date_str -> [comment_text1, comment_text2, ...]}
    """
    daily_comments = {}
    seen_dates = set()
    
    # Check for checkpoint
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{os.path.basename(file_path)}_comments.pkl")
    if os.path.exists(checkpoint_file):
        try:
            logger.info(f"Loading checkpoint from {checkpoint_file}")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                daily_comments = checkpoint_data.get('daily_comments', {})
                seen_dates = checkpoint_data.get('seen_dates', set())
                
            logger.info(f"Loaded {len(daily_comments)} dates and {len(seen_dates)} seen dates from checkpoint")
            
            # Check if we can skip processing
            filled = sum(1 for d in seen_dates if d in daily_comments and len(daily_comments[d]) >= max_comments)
            if seen_dates and (filled / len(seen_dates)) >= early_exit_threshold:
                logger.info(f"Skipping {file_path} as checkpoint data meets exit threshold")
                return daily_comments
        except Exception as e:
            logger.warning(f"Could not load checkpoint, starting fresh: {e}")
    
    try:
        total_chunks = count_chunks(file_path, chunksize)
        if total_chunks == 0:
            logger.error(f"No chunks found in {file_path}")
            return daily_comments
            
        chunk_iter = pd.read_json(file_path, lines=True, chunksize=chunksize)
        
        for i, chunk in enumerate(tqdm(chunk_iter, total=total_chunks, desc=f"Processing {os.path.basename(file_path)}", unit="chunk")):
            try:
                chunk['publishedAt'] = pd.to_datetime(chunk['publishedAt'], errors='coerce')
                chunk = chunk.dropna(subset=['publishedAt'])
                chunk['effective_date'] = chunk['publishedAt'].apply(effective_date).dt.strftime("%Y-%m-%d")
                chunk = chunk[(chunk['effective_date'] >= START_DATE) & (chunk['effective_date'] <= END_DATE)]
                
                for _, row in chunk.iterrows():
                    eff_date = row['effective_date']
                    seen_dates.add(eff_date)
                    
                    # Get text, handle missing values
                    text = row.get("text", "")
                    if not isinstance(text, str) or len(text.strip()) < 10:
                        continue
                        
                    if eff_date not in daily_comments:
                        daily_comments[eff_date] = []
                    
                    if len(daily_comments[eff_date]) < max_comments:
                        # Basic text cleaning
                        text = text.replace('\n', ' ').replace('\r', ' ')
                        daily_comments[eff_date].append(text)
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                continue
            
            # Save checkpoint every 10 chunks
            if i % 10 == 0:
                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({
                            'daily_comments': daily_comments,
                            'seen_dates': seen_dates
                        }, f)
                    logger.info(f"Saved checkpoint after chunk {i}")
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint: {e}")
            
            # Early exit check
            filled = sum(1 for d in seen_dates if d in daily_comments and len(daily_comments[d]) >= max_comments)
            if seen_dates and (filled / len(seen_dates)) >= early_exit_threshold:
                logger.info(f"Early exit: {filled}/{len(seen_dates)} dates filled.")
                break
                
        # Final checkpoint
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'daily_comments': daily_comments,
                    'seen_dates': seen_dates
                }, f)
            logger.info(f"Saved final checkpoint for {file_path}")
        except Exception as e:
            logger.warning(f"Failed to save final checkpoint: {e}")
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
    
    return daily_comments

# -------------------------------
# FinBERT Sentiment Analysis
# -------------------------------
def run_sentiment_analysis(daily_comments):
    """
    Given daily_comments dict, load FinBERT and compute average financial sentiment for each date.
    FinBERT is better suited for financial sentiment than generic sentiment models.
    """
    # Use FinBERT for financial sentiment analysis
    model_name = "ProsusAI/finbert"
    
    try:
        logger.info(f"Loading {model_name} model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        sentiment_model.to(device)
        sentiment_model.eval()
        
        # Use mixed precision if on CUDA
        use_amp = device.type == 'cuda' and PRECISION == "fp16"
        
        # Free memory
        torch.cuda.empty_cache()
        gc.collect()
        
        def analyze_batch(texts, batch_size=BATCH_SIZE):
            if not texts:
                return np.nan
                
            scores = []
            total = len(texts)
            
            for i in range(0, total, batch_size):
                batch = texts[i:i+batch_size]
                try:
                    # Tokenize with padding and truncation
                    encoding = tokenizer(batch, return_tensors="pt", padding=True, 
                                        truncation=True, max_length=512)
                    encoding = {k: v.to(device) for k, v in encoding.items()}
                    
                    # Use mixed precision for inference
                    with torch.no_grad():
                        if use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = sentiment_model(**encoding)
                        else:
                            outputs = sentiment_model(**encoding)
                    
                    # FinBERT returns [negative, neutral, positive]
                    probs = torch.softmax(outputs.logits, dim=1)
                    
                    # Extract sentiment scores: positive - negative for -1 to 1 range
                    batch_scores = probs[:, 2].detach().cpu().numpy() - probs[:, 0].detach().cpu().numpy()
                    scores.extend(batch_scores)
                    
                except Exception as e:
                    logger.error(f"Error in sentiment batch analysis: {e}")
                    continue
                    
                # Free GPU memory after each batch
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                        
            if not scores:
                return np.nan
                
            return np.mean(scores)
        
        # Process each date with progress tracking
        records = []
        checkpoint_file = os.path.join(CHECKPOINT_DIR, "sentiment_analysis_progress.pkl")
        
        # Load checkpoint if exists
        completed_dates = set()
        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    records = checkpoint_data.get('records', [])
                    completed_dates = {r['date'] for r in records}
                logger.info(f"Loaded {len(records)} dates from sentiment checkpoint")
            except Exception as e:
                logger.warning(f"Could not load sentiment checkpoint: {e}")
        
        # Track time to estimate completion
        dates_to_process = [date for date in daily_comments.keys() if date not in completed_dates]
        total_dates = len(dates_to_process)
        
        if total_dates == 0:
            logger.info("All dates already processed for sentiment")
            return pd.DataFrame(records)
            
        logger.info(f"Processing sentiment for {total_dates} dates")
        
        start_time = time.time()
        processed = 0
        
        for date_str in tqdm(dates_to_process, desc="Analyzing daily sentiment"):
            texts = daily_comments[date_str]
            avg_s = analyze_batch(texts)
            records.append({"date": date_str, "avg_sentiment": avg_s})
            
            processed += 1
            
            # Save checkpoint every 20 dates
            if processed % 20 == 0:
                try:
                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'records': records}, f)
                        
                    # Estimate time remaining
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    remaining = (total_dates - processed) / rate if rate > 0 else 0
                    
                    logger.info(f"Processed {processed}/{total_dates} dates. "
                               f"Est. time remaining: {remaining/60:.1f} min")
                except Exception as e:
                    logger.warning(f"Failed to save sentiment checkpoint: {e}")
        
        # Final checkpoint
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'records': records}, f)
        except Exception as e:
            logger.warning(f"Failed to save final sentiment checkpoint: {e}")
            
        return pd.DataFrame(records)
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis module: {e}")
        return pd.DataFrame(columns=["date", "avg_sentiment"])

# -------------------------------
# Merge Multiple News Sources
# -------------------------------
def merge_daily_comments(sources):
    """Merge comments from multiple sources, balancing per date"""
    merged = {}
    
    for source_dict in sources:
        for date, comments in source_dict.items():
            if date not in merged:
                merged[date] = []
            merged[date].extend(comments)
    
    # Ensure we don't exceed MAX_COMMENTS_PER_DAY total
    for date in merged:
        if len(merged[date]) > MAX_COMMENTS_PER_DAY:
            # Randomly sample to get desired count
            merged[date] = np.random.choice(merged[date], MAX_COMMENTS_PER_DAY, replace=False).tolist()
    
    return merged

# -------------------------------
# Main Preprocessing
# -------------------------------
def main_preprocess():
    script_start = time.time()
    logger.info("Starting preprocessing pipeline...")
    
    # Process all news sources
    logger.info("Processing Fox News comments...")
    fox_comments = process_file_chunks(FOX_PATH)
    logger.info(f"Extracted comments for {len(fox_comments)} days from Fox News")
    
    logger.info("Processing CNN comments...")
    cnn_comments = process_file_chunks(CNN_PATH)
    logger.info(f"Extracted comments for {len(cnn_comments)} days from CNN")
    
    logger.info("Processing MSNBC comments...")
    msnbc_comments = process_file_chunks(MSNBC_PATH)
    logger.info(f"Extracted comments for {len(msnbc_comments)} days from MSNBC")
    
    # Merge comments from all sources
    logger.info("Merging comments from all sources...")
    all_comments = merge_daily_comments([fox_comments, cnn_comments, msnbc_comments])
    logger.info(f"Merged data contains {len(all_comments)} unique dates")
    
    total_comments = sum(len(comments) for comments in all_comments.values())
    logger.info(f"Total comments to analyze: {total_comments}")
    
    # Run sentiment analysis
    logger.info("Running FinBERT sentiment analysis...")
    df_sentiment = run_sentiment_analysis(all_comments)
    
    if df_sentiment.empty:
        logger.error("Sentiment analysis failed to produce results")
        return
    
    logger.info(f"Generated sentiment data for {len(df_sentiment)} unique dates")
    logger.info("Sample of daily sentiment:\n" + df_sentiment.head().to_string())
    
    # Merge with stock CSV
    logger.info(f"Loading stock data from {STOCK_DATA_PATH}...")
    try:
        # Save sentiment data separately first
        sentiment_output = "sentiment_data.csv"
        df_sentiment.to_csv(sentiment_output, index=False)
        logger.info(f"Saved raw sentiment data to {sentiment_output}")
        
        # Load stock data
        stock_df = pd.read_csv(STOCK_DATA_PATH)
        
        # Explicitly cast date columns to string for proper matching
        logger.info("Converting date formats for merging...")
        stock_df[STOCK_DATE_COLUMN] = pd.to_datetime(stock_df[STOCK_DATE_COLUMN]).dt.strftime("%Y-%m-%d")
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.strftime("%Y-%m-%d")
        
        # Ensure the sentiment column is float type
        df_sentiment['avg_sentiment'] = df_sentiment['avg_sentiment'].astype(float)
        
        # Debug information
        logger.info(f"Stock date column type: {type(stock_df[STOCK_DATE_COLUMN][0])}")
        logger.info(f"Sentiment date column type: {type(df_sentiment['date'][0])}")
        logger.info(f"Stock date sample: {stock_df[STOCK_DATE_COLUMN].head(3).tolist()}")
        logger.info(f"Sentiment date sample: {df_sentiment['date'].head(3).tolist()}")
        
        # Perform the merge with explicit parameters
        logger.info("Merging sentiment data with stock data...")
        merged_df = pd.merge(
            stock_df, 
            df_sentiment,
            left_on=STOCK_DATE_COLUMN,
            right_on="date",
            how="left"
        )
        
        # Drop duplicate date column
        if "date" in merged_df.columns:
            merged_df.drop(columns=["date"], inplace=True)
            
        # Sort by date
        merged_df.sort_values(STOCK_DATE_COLUMN, inplace=True)
        
        # Manual forward fill for sentiment values
        logger.info("Filling missing sentiment values...")
        
        # Convert to proper numeric type first
        merged_df["avg_sentiment"] = pd.to_numeric(merged_df["avg_sentiment"], errors='coerce')
        
        # Use custom forward fill implementation instead of fillna(method='ffill')
        last_valid_sentiment = 0.0  # Default neutral sentiment
        sentiment_values = merged_df["avg_sentiment"].values
        for i in range(len(sentiment_values)):
            if pd.isna(sentiment_values[i]):
                sentiment_values[i] = last_valid_sentiment
            else:
                last_valid_sentiment = sentiment_values[i]
        
        merged_df["avg_sentiment"] = sentiment_values
        
        # Add comment count column if missing
        if "comment_count" not in merged_df.columns:
            merged_df["comment_count"] = MAX_COMMENTS_PER_DAY
            
        # Check for any remaining issues
        missing_vals = merged_df.isnull().sum()
        if missing_vals.sum() > 0:
            logger.warning(f"Missing values in final dataset:\n{missing_vals[missing_vals > 0]}")
            # Fill any remaining NaNs with 0
            merged_df = merged_df.fillna(0)
        
        # Save to CSV
        logger.info(f"Saving final dataset to {OUTPUT_CSV}...")
        merged_df.to_csv(OUTPUT_CSV, index=False)
        
        # Also save a timestamped copy for backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"sentiment_stock_data_{timestamp}.csv"
        merged_df.to_csv(backup_path, index=False)
        logger.info(f"Backup saved to {backup_path}")
        logger.info(f"Data successfully merged and saved. Dataset has {len(merged_df)} rows with {merged_df.columns.tolist()} columns.")
        
    except Exception as e:
        logger.error(f"Error in merging or saving data: {e}", exc_info=True)
        
        # Try to recover the sentiment data
        try:
            # Load checkpoint data
            checkpoint_file = os.path.join(CHECKPOINT_DIR, "sentiment_analysis_progress.pkl")
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                records = checkpoint_data.get('records', [])
            
            # Save to CSV as backup
            recovery_df = pd.DataFrame(records)
            recovery_path = f"sentiment_data_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            recovery_df.to_csv(recovery_path, index=False)
            logger.info(f"Saved sentiment data recovery file to {recovery_path}")
        except Exception as recovery_error:
            logger.error(f"Failed to recover sentiment data: {recovery_error}")
    
    # Report total runtime
    total_time = (time.time() - script_start) / 60
    logger.info(f"Total processing time: {total_time:.2f} minutes")
    
    # Check if within time budget
    if total_time < 300:  # 5 hours = 300 minutes
        logger.info("Processing completed within the 5-hour time budget")
    else:
        logger.warning(f"Processing exceeded the 5-hour time budget: {total_time:.2f} minutes")

if __name__ == "__main__":
    try:
        main_preprocess()
    except Exception as e:
        logger.critical(f"Fatal error in main preprocessing pipeline: {e}", exc_info=True)
