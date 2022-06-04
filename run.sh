echo "----- source 0, target 1 -----"
python main.py --mode s2t --source 0 --target 1 --seed 2024 | tee shot_s0_t1.txt
echo "----- source 0, target 2 -----"
python main.py --mode s2t --source 0 --target 2 --seed 2024 | tee shot_s0_t2.txt
echo "----- source 0, target 3 -----"
python main.py --mode s2t --source 0 --target 3 --seed 2024 | tee shot_s0_t3.txt
echo "----- source 1, target 0 -----"
python main.py --mode s2t --source 1 --target 0 --seed 2025 | tee shot_s1_t0.txt
echo "----- source 1, target 2 -----"
python main.py --mode s2t --source 1 --target 2 --seed 2025 | tee shot_s1_t2.txt
echo "----- source 1, target 3 -----"
python main.py --mode s2t --source 1 --target 3 --seed 2025 | tee shot_s1_t3.txt
echo "----- source 2, target 0 -----"
python main.py --mode s2t --source 2 --target 0 --seed 2026 | tee shot_s2_t0.txt
echo "----- source 2, target 1 -----"
python main.py --mode s2t --source 2 --target 1 --seed 2026 | tee shot_s2_t1.txt
echo "----- source 2, target 3 -----"
python main.py --mode s2t --source 2 --target 3 --seed 2026 | tee shot_s2_t3.txt
echo "----- source 3, target 0 -----"
python main.py --mode s2t --source 3 --target 0 --seed 2027 | tee shot_s3_t0.txt
echo "----- source 3, target 1 -----"
python main.py --mode s2t --source 3 --target 1 --seed 2027 | tee shot_s3_t1.txt
echo "----- source 3, target 2 -----"
python main.py --mode s2t --source 3 --target 2 --seed 2027 | tee shot_s3_t2.txt
