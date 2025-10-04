data "aws_s3_bucket" "data" {
  for_each = toset(["hrrr", "mrms"])
  bucket   = "${each.value}-${var.environment}-bucket"
}

data "aws_s3_bucket" "trained-dl-models" {
  bucket = "trained-dl-models"
}
