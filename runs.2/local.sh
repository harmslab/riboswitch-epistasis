for g in `echo "UxxGx Uxxxx xCxGx xCxxx xxAGx xxAxx xxxGx xxxxx"`; do
    for m in `echo "2.3 3.3 3.4 4.4 4.5"`; do
        echo "$g ml $m"
        python run.py $g ml $m
    done
done
