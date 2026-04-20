#!/bin/bash
mkdir -p /mnt/e/5g_handover/phase_3/test_dataset
cd /home/arjyadeep/ns-3-dev

echo "Generating Scenario 1..."
build/scratch/ns3-dev-ns3_md_scenarios-optimized --scenarioId=1 --pattern=A --ueCount=10 --duration=300 --seed=999 --outputPrefix=/mnt/e/5g_handover/phase_3/test_dataset/s1_A

echo "Generating Scenario 4..."
build/scratch/ns3-dev-ns3_md_scenarios-optimized --scenarioId=4 --pattern=B --ueCount=10 --duration=300 --seed=999 --outputPrefix=/mnt/e/5g_handover/phase_3/test_dataset/s4_B

echo "Generating Scenario 7..."
build/scratch/ns3-dev-ns3_md_scenarios-optimized --scenarioId=7 --pattern=A --ueCount=10 --duration=300 --seed=999 --outputPrefix=/mnt/e/5g_handover/phase_3/test_dataset/s7_A
