resource "aws_instance" "hail-inference1" {

  instance_type               = "g4dn.xlarge"
  ami                         = "ami-0487b2b453ee31904" # Deep Learning AMI GPU CUDA 11.4.3 
  subnet_id                   = data.terraform_remote_state.django_environment.outputs.vpc.public_subnets[0]
  vpc_security_group_ids      = [aws_security_group.sg_flash_dl.id]
  associate_public_ip_address = true
  key_name                    = aws_key_pair.ec2_default_key.key_name
  iam_instance_profile        = aws_iam_instance_profile.inference1.name
  tags = {
    Name = "hail-ec2-inference1-${local.app_env}"
  }
  user_data = <<EOF
#!/bin/bash
sudo apt-get update -y
sudo mkdir /data
sudo mkdir /usr/local/inference1_methods

docker pull nvcr.io/nvidia/tensorrt:22.08-py3
sudo apt install -y awscli

sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose

aws s3 sync s3://${aws_s3_bucket.method_staging.id}/inference1/ /usr/local/inference1_methods/.

# lookup buckets & set env vars
echo "REDIS_URL_0=${aws_elasticache_cluster.redis0.cache_nodes[0].address}" > /usr/local/inference1_methods/.env
echo "REDIS_URL_1=${aws_elasticache_cluster.redis1.cache_nodes[0].address}" >> /usr/local/inference1_methods/.env
echo "OUTPUT_BUCKET=${aws_s3_bucket.hail-output.id}" >> /usr/local/inference1_methods/.env
echo "OUTPUT_ARCHIVE_BUCKET=${aws_s3_bucket.hail-output-archive.id}" >> /usr/local/inference1_methods/.env
echo "MODEL_BUCKET=${data.aws_s3_bucket.trained-dl-models.id}" >> /usr/local/inference1_methods/.env
echo "REDIS_PORT=6379" >> /usr/local/inference1_methods/.env

curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update

sudo apt-get install -y nvidia-container-runtime

cat << DAEMON > /etc/docker/daemon.json
{
  "runtimes": {
      "nvidia": {
          "path": "/usr/bin/nvidia-container-runtime",
          "runtimeArgs": []
      }
  },
  "default-runtime": "nvidia"
}
DAEMON

sudo systemctl restart docker

cd /usr/local/inference1_methods/
touch predict_log.txt
sleep 7m
docker-compose up
EOF


  root_block_device {
    volume_size = var.volume_size_inference
    encrypted   = true
  }
}

resource "aws_iam_instance_profile" "inference1" {
  name = "inference1-${local.app_env}"
  role = aws_iam_role.inference1.name
}
