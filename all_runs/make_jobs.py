if __name__ == "__main__":
    count = 0
    with open("submit_and_run.sh","w") as super_f:
        super_f.write("#! /bin/bash\n")
        for rs in [1,2,3,4,5]:
            for level in ["L1", "L2"]:
                for splitting in ["spatial", "species"]:
                    for embedding in ["doc2vec", "longformer"]:
                        for merge_method in ["fusion","selection"]:
                            count += 1
                            with open(f'jobs/job{count}.run', 'w') as f:
                                f.write("#!/bin/bash -l")
                                f.write("\n")
                                f.write("\n")
                                f.write("#SBATCH --nodes 1\n#SBATCH --ntasks 1\n#SBATCH --cpus-per-task 16\n#SBATCH --mem 16G\n#SBATCH --time 00:20:00\n#SBATCH --partition=gpu\n#SBATCH --qos=gpu\n#SBATCH --gres=gpu:1")
                                f.write("\n")
                                for run_mode in ["train","test"]:
                                    f.write("\n")
                                    f.write("python run.py ")
                                    f.write(f"--RANDOM_STATE {rs} ")
                                    f.write(f"--EMBEDDING {embedding} ")
                                    f.write(f"--LEVEL {level} ")
                                    f.write(f"--SPLITTING {splitting} ")
                                    f.write(f"--MERGE_METHOD {merge_method} ")
                                    f.write(f"--RUN_MODE {run_mode} ")
                                    f.write(f"--JOB_ID {count} ")
                                    a = "yes" if run_mode == "test" else "no"
                                    f.write(f"--LOAD_CHECKPOINT {a} ")
                            super_f.write(f"chmod +x jobs/job{count}.run\n")
                            super_f.write(f"sbatch jobs/job{count}.run\n")
                            super_f.write("\n")
                    embedding = "agnostic"
                    count += 1
                    with open(f'jobs/job{count}.run', 'w') as f:
                        f.write("#!/bin/bash -l")
                        f.write("\n")
                        f.write("\n")
                        f.write("#SBATCH --nodes 1\n#SBATCH --ntasks 1\n#SBATCH --cpus-per-task 16\n#SBATCH --mem 16G\n#SBATCH --time 00:20:00\n#SBATCH --partition=gpu\n#SBATCH --qos=gpu\n#SBATCH --gres=gpu:1")
                        f.write("\n")
                        for run_mode in ["train","test"]:
                            f.write("\n")
                            f.write("python run.py ")
                            f.write(f"--RANDOM_STATE {rs} ")
                            f.write(f"--EMBEDDING {embedding} ")
                            f.write(f"--LEVEL {level} ")
                            f.write(f"--SPLITTING {splitting} ")
                            f.write(f"--RUN_MODE {run_mode} ")
                            f.write(f"--JOB_ID {count} ")
                            a = "yes" if run_mode == "test" else "no"
                            f.write(f"--LOAD_CHECKPOINT {a} ")
                    super_f.write(f"chmod +x jobs/job{count}.run\n")
                    super_f.write(f"sbatch jobs/job{count}.run\n")
                    super_f.write("\n")

            for fraction in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                for embedding in ["doc2vec", "longformer"]:
                    merge_method = "fusion"
                    count += 1
                    with open(f'jobs/job{count}.run', 'w') as f:
                        f.write("#!/bin/bash -l")
                        f.write("\n")
                        f.write("\n")
                        f.write("#SBATCH --nodes 1\n#SBATCH --ntasks 1\n#SBATCH --cpus-per-task 16\n#SBATCH --mem 16G\n#SBATCH --time 00:20:00\n#SBATCH --partition=gpu\n#SBATCH --qos=gpu\n#SBATCH --gres=gpu:1")
                        f.write("\n")
                        for run_mode in ["train","test"]:
                            f.write("\n")
                            f.write("python run.py ")
                            f.write(f"--RANDOM_STATE {rs} ")
                            f.write(f"--EMBEDDING {embedding} ")
                            f.write(f"--LEVEL L1 ")
                            f.write(f"--SPLITTING progressive ")
                            f.write(f"--FRACTION {fraction} ")
                            f.write(f"--MERGE_METHOD {merge_method} ")
                            f.write(f"--RUN_MODE {run_mode} ")
                            f.write(f"--JOB_ID {count} ")
                            a = "yes" if run_mode == "test" else "no"
                            f.write(f"--LOAD_CHECKPOINT {a} ")
                    super_f.write(f"chmod +x jobs/job{count}.run\n")
                    super_f.write(f"sbatch jobs/job{count}.run\n")
                    super_f.write("\n")
                embedding = "agnostic"
                count += 1
                with open(f'jobs/job{count}.run', 'w') as f:
                    f.write("#!/bin/bash -l")
                    f.write("\n")
                    f.write("\n")
                    f.write("#SBATCH --nodes 1\n#SBATCH --ntasks 1\n#SBATCH --cpus-per-task 16\n#SBATCH --mem 16G\n#SBATCH --time 00:20:00\n#SBATCH --partition=gpu\n#SBATCH --qos=gpu\n#SBATCH --gres=gpu:1")
                    f.write("\n")
                    for run_mode in ["train","test"]:
                        f.write("\n")
                        f.write("python run.py ")
                        f.write(f"--RANDOM_STATE {rs} ")
                        f.write(f"--EMBEDDING {embedding} ")
                        f.write(f"--LEVEL L1 ")
                        f.write(f"--SPLITTING progressive ")
                        f.write(f"--FRACTION {fraction} ")
                        f.write(f"--RUN_MODE {run_mode} ")
                        f.write(f"--JOB_ID {count} ")
                        a = "yes" if run_mode == "test" else "no"
                        f.write(f"--LOAD_CHECKPOINT {a} ")
                super_f.write(f"chmod +x jobs/job{count}.run\n")
                super_f.write(f"sbatch jobs/job{count}.run\n")
                super_f.write("\n")
