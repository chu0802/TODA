echo "----- source 0 -----"
python main.py --mode 3shot --source 0 --target 1 --seed 2020
echo "----- source 1 -----"
python main.py --mode 3shot --source 1 --target 0 --seed 2023
echo "----- source 2 -----"
python main.py --mode 3shot --source 2 --target 1 --seed 2026
echo "----- source 3 -----"
python main.py --mode 3shot --source 3 --target 1 --seed 2029
# echo "----- source 0, target 2 -----"
# python main.py --mode 3shot --source 0 --target 2 --seed 2021 | tee 3shot_s0_t2_2021.txt
# echo "----- source 0, target 3 -----"
# python main.py --mode 3shot --source 0 --target 3 --seed 2022 | tee 3shot_s0_t3_2022.txt
# echo "----- source 1, target 0 -----"
# python main.py --mode 3shot --source 1 --target 0 --seed 2023 | tee 3shot_s1_t0_2023.txt
# echo "----- source 1, target 2 -----"
# python main.py --mode 3shot --source 1 --target 2 --seed 2024 | tee 3shot_s1_t2_2024.txt
# echo "----- source 1, target 3 -----"
# python main.py --mode 3shot --source 1 --target 3 --seed 2025 | tee 3shot_s1_t3_2025.txt
# echo "----- source 2, target 0 -----"
# python main.py --mode 3shot --source 2 --target 0 --seed 2026 | tee 3shot_s2_t0_2026.txt
# echo "----- source 2, target 1 -----"
# python main.py --mode 3shot --source 2 --target 1 --seed 2027 | tee 3shot_s2_t1_2027.txt
# echo "----- source 2, target 3 -----"
# python main.py --mode 3shot --source 2 --target 3 --seed 2028 | tee 3shot_s2_t3_2028.txt
# echo "----- source 3, target 0 -----"
# python main.py --mode 3shot --source 3 --target 0 --seed 2029 | tee 3shot_s3_t0_2029.txt
# echo "----- source 3, target 1 -----"
# python main.py --mode 3shot --source 3 --target 1 --seed 2030 | tee 3shot_s3_t1_2030.txt
# echo "----- source 3, target 2 -----"
# python main.py --mode 3shot --source 3 --target 2 --seed 2031 | tee 3shot_s3_t2_2031.txt
