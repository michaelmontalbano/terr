resource "aws_iam_policy" "inference1" {
  name   = "inference1-${local.app_env}"
  policy = <<EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "ViewAccountPasswordRequirements",
            "Effect": "Allow",
            "Action": [
              "s3:*"
            ],
            "Resource": [
                "${aws_s3_bucket.hail-output.arn}", 
                "${aws_s3_bucket.hail-output.arn}/*",
                "${aws_s3_bucket.method_staging.arn}/*", 
                "${aws_s3_bucket.method_staging.arn}",
                "${data.aws_s3_bucket.trained-dl-models.arn}/*"
            ]
        }
    ]
}
EOF
}

resource "aws_iam_role" "inference1" {
  name               = "inference1-${local.app_env}"
  assume_role_policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "inference1" {
  role       = aws_iam_role.inference1.name
  policy_arn = aws_iam_policy.inference1.arn
}
