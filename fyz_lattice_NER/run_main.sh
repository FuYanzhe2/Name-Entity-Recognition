#python main.py --status train \
#		--train ../data/onto4ner.cn/train.char.bmes \
#		--dev ../data/onto4ner.cn/dev.char.bmes \
#		--test ../data/onto4ner.cn/test.char.bmes \
#		--savemodel ../data/onto4ner.cn/saved_model \


python3 main.py --status decode \
		--raw ./ResumeNER/test.char.bmes \
		--savedset ./ResumeNER/save.dset \
		--loadmodel ./data/model/saved_model2.lstmcrf..36.model \
		--output ./ResumeNER/raw.out \
