# create bucket
resource "aws_s3_bucket" "method_staging" {
  bucket = "${local.app_env}-combination-methods"
}

resource "aws_s3_bucket_policy" "method_staging" {
  bucket = aws_s3_bucket.method_staging.id
  policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Statement1",
            "Effect": "Allow",
            "Principal": {
                "AWS": [
                  "arn:aws:iam::869179685131:role/ecstask-flash-sandbox",
                  "arn:aws:iam::869179685131:role/flash-ingest-monitoring-sandbox"
                ]
            },
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:PutObjectAcl",
                "s3:GetObjectAcl",
                "s3:ListBucket"
            ],
            "Resource": [
                "${aws_s3_bucket.method_staging.arn}/*",
                "${aws_s3_bucket.method_staging.arn}"
            ]
        }
    ]
}
EOF
}

resource "aws_s3_bucket_object" "combination" {
  for_each = toset([
    "data_combiner.py",
    "data_prep_scales.py",
    "dual_2.py",
    "dbz50_sum.npy",
    "global_mins.npy",
    "global_maxs.npy",
    "scaling_utils.py"
  ])
  bucket      = aws_s3_bucket.method_staging.id
  key         = "combination/${each.value}"
  source      = "../../combination/${each.value}"
  source_hash = filemd5("../../combination/${each.value}")
}
resource "aws_s3_bucket_object" "inference1" {
  for_each = toset([
    "model.h5",
    "Dockerfile",
    "docker-compose.yml",
    "rnn_intervals.keras",
    "predict.py",
    "requirements.txt",
    "rnn.py",
    "models.py",
    "config.py"
  ])
  bucket      = aws_s3_bucket.method_staging.id
  key         = "inference1/${each.value}"
  source      = "../../inference1/${each.value}"
  source_hash = filemd5("../../inference1/${each.value}")
}

resource "aws_s3_bucket_object" "output" {
  for_each = toset([
    "alert.py",
    "create_geojson.py",
    "helpers.py",
    "process_raw_forecast.py",
    "lats.npy",
    "lons.npy",
    "thresholds_extended.pickle",
    "thresholds.pickle",
    "psuedo_customer_fixtures.csv",
    "all_customers.csv",
    "run_output_loop.sh",
    "best_pred_for_obs_by_lead_20250923.csv",
    "verify.py"
  ])
  bucket      = aws_s3_bucket.method_staging.id
  key         = "output/${each.value}"
  source      = "../../output/${each.value}"
  source_hash = filemd5("../../output/${each.value}")
}
