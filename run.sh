confidenceLevel="0.99 0.95 0"
for i in $confidenceLevel; do
    python decisiontree.py training.txt validation.txt $i
done