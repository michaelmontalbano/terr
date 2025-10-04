import os
import sys
import pickle
import time
import json
from datetime import datetime
from pathlib import Path

# Ensure repository root is importable so evaluation loaders can resolve shared modules
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import boto3
import redis
import numpy as np
from dotenv import load_dotenv

from config import EvaluationConfig
from models import ModelLoader

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predict.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def enhanced_load_data_chunked(redis_client, key):
    """Load chunked data from Redis"""
    try:
        meta_key = f"{key}_meta"
        meta_raw = redis_client.get(meta_key)
        
        if meta_raw is None:
            logger.warning(f"No metadata found for '{key}'")
            return None
        
        meta = pickle.loads(meta_raw)
        logger.info(f"Loading {meta['num_chunks']} chunks, {meta['total_size']} bytes total")
        
        # Load chunks
        compressed_data = b""
        for i in range(meta['num_chunks']):
            chunk_key = f"{key}_chunk_{i}"
            chunk = redis_client.get(chunk_key)
            if chunk is None:
                raise ValueError(f"Missing chunk {i}")
            compressed_data += chunk
        
        # Decompress
        if meta.get('compression') == 'zlib':
            import zlib
            data = pickle.loads(zlib.decompress(compressed_data))
        else:
            data = pickle.loads(compressed_data)
        
        logger.info(f"Loaded data: shape={data.shape}, dtype={data.dtype}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load chunked data: {e}")
        return None

def wait_until_slot(target_minutes=5):
    """Wait until next N-minute boundary"""
    now = datetime.utcnow()
    wait_seconds = (target_minutes - now.minute % target_minutes) * 60 - now.second
    if wait_seconds > 0:
        logger.info(f"Waiting {wait_seconds}s until {target_minutes}-min boundary")
        time.sleep(wait_seconds)

# Load environment
load_dotenv()
redis_url = os.getenv("REDIS_URL_0")
output_bucket = os.getenv("OUTPUT_BUCKET")

# Configuration for lead time
LEAD_TIME_INDEX = int(os.getenv("LEAD_TIME_INDEX", "4"))  # Default to 5th prediction (index 4 = 25 minutes)
LEAD_TIME_MINUTES = (LEAD_TIME_INDEX + 1) * 5  # Convert index to minutes

logger.info(f"Starting at {datetime.utcnow()}")
logger.info(f"Redis: {redis_url}, Output: {output_bucket}")
logger.info(f"Using lead time index {LEAD_TIME_INDEX} ({LEAD_TIME_MINUTES} minutes)")

# Connect to services
s3 = boto3.client("s3")
r = redis.Redis(host=redis_url, port=6379, db=0)
assert r.ping(), "Cannot connect to Redis"

# Resolve model path (allow overriding via environment variable)
env_model_path = os.getenv("MODEL_PATH")
if env_model_path:
    model_path = Path(env_model_path)
    if not model_path.is_absolute():
        model_path = (ROOT_DIR / model_path).resolve()
else:
    model_path = (Path(__file__).parent / "model.h5").resolve()

model_type = os.getenv("MODEL_TYPE", "convgru")
config = EvaluationConfig(model_path=str(model_path), model_type=model_type)
loader = ModelLoader()
loaded_model = loader.load(config)
model = loaded_model.model
logger.info("Model loaded successfully as %s from %s", loaded_model.model_type.value, model_path)

# Main loop
while True:
    cycle_start = time.time()
    
    # Load data
    data = enhanced_load_data_chunked(r, "data")
    if data is None:
        logger.warning("No data available, waiting 30s...")
        time.sleep(30)
        continue
    
    load_time = time.time() - cycle_start
    
    # Sync to 5-minute boundary (accounting for load time)
    wait_until_slot(5)
    
    logger.info(f"="*60)
    logger.info(f"Starting predictions at {datetime.utcnow().strftime('%H:%M:%S')}")
    
    # Generate predictions
    predict_start = time.time()
    sz = 256
    width = sz // 2
    data_height, data_width = data.shape[1], data.shape[2]
    
    # Initialize grid for single lead time
    full_prediction = np.zeros((data_height, data_width), dtype=np.float32)
    count_grid = np.zeros((data_height, data_width), dtype=np.float32)
    
    pairs = []
    total_tiles = 0
    
    for x in range(width + sz//4, data_width - width - sz//4, width):
        for y in range(width + sz//4, data_height - width - sz//4, width):
            if y + sz <= data_height and x + sz <= data_width:
                # Extract tile
                tile = data[:, y:y+sz, x:x+sz, :]
                
                # IMPORTANT: Reverse time dimension to match evaluation
                tile = tile[::-1, :, :, :]
                
                # Expand dims and predict
                tile_expanded = np.expand_dims(tile, axis=0)
                pred = model.predict(tile_expanded, batch_size=1, verbose=0)
                
                # Extract only the specified lead time prediction
                # pred shape: (1, 12, 256, 256, 1)
                pred_at_lead = pred[0, LEAD_TIME_INDEX, :, :, 0]  # Extract (256, 256) for specific lead time
                
                # Accumulate predictions
                full_prediction[y:y+sz, x:x+sz] += pred_at_lead
                count_grid[y:y+sz, x:x+sz] += 1
                
                pairs.append((y, x))
                total_tiles += 1
    
    predict_time = time.time() - predict_start
    logger.info(f"Generated {total_tiles} predictions in {predict_time:.1f}s for {LEAD_TIME_MINUTES}-minute lead time")
    
    if total_tiles > 0:
        # Average overlapping regions
        mask = count_grid > 0
        full_prediction[mask] /= count_grid[mask]
        
        # Get timestamp
        mrms_files_raw = r.get("mrms_files")
        if mrms_files_raw:
            mrms_files = np.frombuffer(mrms_files_raw, "<U20")
            folder_name = str(mrms_files[-1]) if len(mrms_files) > 0 else f"mrms_{datetime.utcnow().strftime('%Y%m%dT%H:%MZ')}"
        else:
            folder_name = f"mrms_{datetime.utcnow().strftime('%Y%m%dT%H:%MZ')}"
        
        # Save single lead time prediction
        model_name = os.getenv("MODEL_NAME", "model_1")
        
        # Option 1: Save as single array with metadata
        output_data = {
            'prediction': full_prediction,
            'lead_time_minutes': LEAD_TIME_MINUTES,
            'lead_time_index': LEAD_TIME_INDEX,
            'timestamp': datetime.utcnow().isoformat(),
            'shape': full_prediction.shape
        }
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{folder_name}/{model_name}_lead{LEAD_TIME_MINUTES}_output.pkl",
            Body=pickle.dumps(output_data)
        )
        
        logger.info(f"Saved {LEAD_TIME_MINUTES}-minute prediction to {folder_name}/{model_name}_lead{LEAD_TIME_MINUTES}_output.pkl")
        
        # Option 2: Also save in legacy format for compatibility (single timestep in dict)
        legacy_forecast_data = {str(LEAD_TIME_MINUTES): full_prediction}
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"{folder_name}/{model_name}_output.pkl",
            Body=pickle.dumps(legacy_forecast_data)
        )
        
        # Save metrics
        cycle_time = time.time() - cycle_start
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cycle_time': cycle_time,
            'load_time': load_time,
            'predict_time': predict_time,
            'tiles': total_tiles,
            'lead_time_minutes': LEAD_TIME_MINUTES,
            'lead_time_index': LEAD_TIME_INDEX,
            'prediction_shape': list(full_prediction.shape),
            'prediction_stats': {
                'min': float(np.min(full_prediction)),
                'max': float(np.max(full_prediction)),
                'mean': float(np.mean(full_prediction)),
                'std': float(np.std(full_prediction))
            }
        }
        
        s3.put_object(
            Bucket=output_bucket,
            Key=f"metrics/{datetime.utcnow().strftime('%Y%m%d')}/{folder_name}.json",
            Body=json.dumps(metrics)
        )
        
        logger.info(f"Cycle complete: {cycle_time:.1f}s total")
        logger.info(f"Prediction stats - Min: {metrics['prediction_stats']['min']:.2f}, "
                   f"Max: {metrics['prediction_stats']['max']:.2f}, "
                   f"Mean: {metrics['prediction_stats']['mean']:.2f}")
    
    # Wait for next cycle
    elapsed = time.time() - cycle_start
    if elapsed < 200:
        time.sleep(200 - elapsed)
    else:
        logger.warning(f"Cycle took {elapsed:.1f}s - exceeded 5 minutes!")
