output=""

for i in {1..20}
do
    echo "Run $i"
    python bow_main.py &> temp_output.txt
    temp=$(cat temp_output.txt | grep -E "test pos sample accuracy:|test neg sample accuracy:")
    output+="$temp\n"
done

echo -e "$output" > output.txt
rm temp_output.txt