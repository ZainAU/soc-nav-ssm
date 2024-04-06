import subprocess

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def main(run_type:  str     = 'test',
         policy:    str     = 'Naive-MambaRL', 
         use_gpu:   str     = 'yes'):
    run_type =  run_type.lower()
    policy   =  policy
    use_gpu  =  use_gpu.lower() == 'yes'

    gpu_option = "--gpu" if use_gpu  else ""

    if run_type == 'test':
        model_dir = input("Enter the model directory: ")
        commands = [
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_bl-cr.config {gpu_option}",
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_bl-sq.config {gpu_option}",
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_dn-cr.config {gpu_option}",
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_dn-sq.config {gpu_option}",
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_lg-cr.config {gpu_option}",
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_lg-sq.config {gpu_option}"
        ]
    elif run_type == 'visualize':
        model_dir = input("Enter the model directory: ")
        setting = input("Enter the setting (bl-cr, bl-sq, dn-cr, dn-sq, lg-cr, lg-sq): ")
        test_case = int(input("Enter the test case number: "))
        traj = input("Do you want to include the --traj tag? (yes/no): ").lower() == 'yes'
        traj_option = "--traj" if traj else ""

        commands = [
            f"python test.py --policy {policy} --model_dir {model_dir} --phase test --env_config configs/env_multi-test_{setting}.config {gpu_option} --visualize --test_case {test_case} {traj_option}"
        ]
    elif run_type == 'train':
        output_dir = input("Enter the output directory: ")
        config = input("Enter the env config: ")
        commands = [
            f"python train.py --policy {policy} --output_dir data/{output_dir} --env_config configs/{config}.config {gpu_option}"
        ]

    else:
        print("Invalid option. Please choose 'test all' or 'visualize'.")
        return

    for command in commands:
        if not run_command(command):
            print("Command execution failed.")
            break

if __name__ == "__main__":
    main()
