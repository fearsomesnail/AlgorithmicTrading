terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Secure random RDS password (used if var.db_password not provided)
resource "random_password" "db" {
  length  = 24
  special = true
}

# Import/create a Key Pair from local public key for bastion
resource "aws_key_pair" "bastion" {
  key_name   = "${var.name_prefix}-bastion-key"
  public_key = file("~/.ssh/bastion_sshKP.pub")
}

module "vpc" {
  source = "./modules/vpc"

  name_prefix       = var.name_prefix
  vpc_cidr          = var.vpc_cidr
  public_cidrs      = var.public_cidrs
  private_cidrs     = var.private_cidrs
  enable_nat_gateway = true
}

module "security" {
  source = "./modules/security"

  name_prefix         = var.name_prefix
  vpc_id              = module.vpc.vpc_id
  allowed_ssh_cidr    = var.allowed_ssh_cidr
}

module "bastion" {
  source = "./modules/bastion"

  name_prefix   = var.name_prefix
  subnet_id     = module.vpc.public_subnet_ids[0]
  vpc_id        = module.vpc.vpc_id
  key_name      = aws_key_pair.bastion.key_name
  instance_type = var.bastion_instance_type
  sg_id         = module.security.bastion_sg_id
}

module "rds" {
  source = "./modules/rds"

  name_prefix             = var.name_prefix
  db_name                 = var.db_name
  db_username             = var.db_username
  db_password             = coalesce(var.db_password, random_password.db.result)
  subnet_ids              = module.vpc.private_subnet_ids
  vpc_id                  = module.vpc.vpc_id
  engine_version          = var.db_engine_version
  instance_class          = var.db_instance_class
  allocated_storage       = var.db_allocated_storage
  max_allocated_storage   = var.db_max_allocated_storage
  multi_az                = var.db_multi_az
  backup_retention_period = var.db_backup_retention_period
  skip_final_snapshot     = var.db_skip_final_snapshot
  deletion_protection     = var.db_deletion_protection
  performance_insights_enabled = var.db_performance_insights_enabled
  bastion_sg_id           = module.security.bastion_sg_id
  private_instance_sg_id  = module.security.private_instance_sg_id
}

module "private_instance" {
  source = "./modules/private_instance"

  name_prefix        = var.name_prefix
  subnet_id          = module.vpc.private_subnet_ids[0]
  sg_id              = module.security.private_instance_sg_id
  key_name           = aws_key_pair.bastion.key_name
  instance_type      = var.private_instance_type
  rds_endpoint       = module.rds.endpoint
  db_password        = random_password.db.result
  github_repo        = "https://github.com/fearsomesnail/ALGOTRADING.git"
  bastion_public_key = file("~/.ssh/bastion_sshKP.pub")
}

module "sagemaker" {
  source = "./modules/sagemaker"

  name_prefix      = var.name_prefix
  subnet_id        = module.vpc.private_subnet_ids[0]
  vpc_id           = module.vpc.vpc_id
  vpc_cidr         = var.vpc_cidr
  aws_region       = var.aws_region
  instance_type    = var.sagemaker_instance_type
  route_table_ids  = [module.vpc.private_route_table_id]
}

output "rds_endpoint" {
  value = module.rds.endpoint
}

output "bastion_public_ip" {
  value = module.bastion.public_ip
}

output "private_instance_ip" {
  value = module.private_instance.private_ip
}

output "sagemaker_notebook_url" {
  value = module.sagemaker.notebook_url
}

output "generated_db_password" {
  value     = coalesce(var.db_password, random_password.db.result)
  sensitive = true
}

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "private_subnet_ids" {
  value = module.vpc.private_subnet_ids
}

output "public_subnet_ids" {
  value = module.vpc.public_subnet_ids
}


