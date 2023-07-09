# data preprocessing for THGAT input

1. preprocess the dataset and change it to THGAT input format:
   
   ```
   step 1: python preprocess_data.py dataset
   example: python preprocess_data.py cora
   
   step 2: python process2THGAT_format.py dataset feature_dim
   example: python process2THGAT_format.py cora 128
   ```

2. generate node signature:
   
   1. node signature based on type prime
      
      ```
      step 1: python gen_prime_map.py dataset
      example: python gen_prime_map.py cora
      
      step 2: python gen_sig_prime.py dataset sig_dim max_bias
      example: python gen_sig_prime.py cora 128 1000
      ```
   
   2. node signature based on type degree
      ```
      command: python gen_sig_degree.py --dataset dataset_name
      example: python gen_sig_degree.py --dataset cora
      ```
      