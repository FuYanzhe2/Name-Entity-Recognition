#!/bin/bash
python3 main.py --status decode \
 		--raw ./ResumeNER/test.char.bmes \
		--savedset ./ResumeNER/save.dset.dset \
		--loadmodel ./data/model/saved_model.lstmcrf..45.model \
 		--output ./ResumeNER/demo.raw.out \
