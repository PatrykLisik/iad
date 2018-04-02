
START=$(date +%s.%N)

#Wykres stanów pośrednich sieci
python3 middleStages.py train1.txt test.txt &
python3 middleStages.py train2.txt test.txt &

#Aproksymacja - wykres dla różnych learnig rate
python3 aproxLr.py train1.txt test.txt &
python3 aproxLr.py train2.txt test.txt &

#Aproksymacja - wykres dla różnych ilości neuronów
python3 aprox.py train1.txt test.txt &
python3 aprox.py train2.txt test.txt &

#Błędy - wykres błedu średnoi kwadratowego od ilości iteracji
python3 errors.py train1.txt test.txt &
python3 errors.py train2.txt test.txt &

#Dane csv do tabel
python3 tabela.py train1.txt test.txt
python3 tabela.py train2.txt test.txt

wait
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo "Ended after $DIFF seconds"
