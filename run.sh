FILE=COVID-19

if [ -d "$FILE" ]; then
    cd COVID-19
    git pull
    cd ..
else
    git clone https://github.com/CSSEGISandData/COVID-19.git
fi

python analysis.py
