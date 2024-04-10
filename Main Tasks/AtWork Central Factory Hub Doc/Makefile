SOURCE = cfh_manual.tex
OUT = cfh_manual
PDFLATEX = pdflatex
  
all: subdirs compile

subdirs:
	mkdir -p ./out

draft: subdirs
	$(PDFLATEX)--draftmode --shell-escape --output-directory=./out --jobname $(OUT) $(SOURCE)
	cp ./out/$(OUT).pdf .
		
once: subdirs
	$(PDFLATEX) --shell-escape --output-directory=./out --jobname $(OUT) $(SOURCE)
	cp ./out/$(OUT).pdf .
		
compile:
	$(PDFLATEX) --shell-escape --output-directory=./out --jobname $(OUT) $(SOURCE)
	$(PDFLATEX) --shell-escape --output-directory=./out --jobname $(OUT) $(SOURCE)
	cp ./out/$(OUT).pdf .

clean:
	rm -rf ./out
