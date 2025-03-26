from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as spark_max, to_timestamp

def get_max_date(jsonl_file):
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("MaxDateFinder") \
        .getOrCreate()
    
    # Read the JSONL file (each line is a JSON record)
    df = spark.read.json(jsonl_file)
    
    # Convert the publishedAt field to a timestamp
    df = df.withColumn("publishedAt_ts", to_timestamp(col("publishedAt")))
    
    # Compute the maximum publishedAt timestamp
    max_date_row = df.agg(spark_max("publishedAt_ts").alias("max_date")).collect()[0]
    max_date = max_date_row["max_date"]
    
    # Stop the Spark session
    spark.stop()
    
    return max_date

if __name__ == "__main__":
    jsonl_file_path = "C:/Users/ankit/Downloads/msnbc-002.jsonl"
    latest_date = get_max_date(jsonl_file_path)
    print("Latest publishedAt date in dataset:", latest_date)
