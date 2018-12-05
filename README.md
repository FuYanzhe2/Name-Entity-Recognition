# Name-Entity-Recognition
Lstm-crf,Lattice-CRF,bert-ner及近年ner相关论文follow

- ChineseNER 中文NER 
	tensorflow 1.4.0
    use method :
        python3 main.py
    
    详细使用原来即实验结果见博客https://www.jianshu.com/p/aed50c1b2930

- fyz_lattice_NER 中文NER lattice model
	pytorch 0.4.0
	Python 3.6
	use method :
        python3 main.py
		或者直接配置然后运行：bash fyz_run_decode.sh
	
	详细使用原来即实验结果见博客https://www.jianshu.com/p/9c99796ff8d9
	文件中需要的两个词向量地址：
		链接：https://pan.baidu.com/s/1Uj97799tGjdET_vbdkW7tQ 
		提取码：vgwi 
		解压文件 放到data/ 文件夹下即可
		
- BERT-BiLSTM-CRF-NER
	tensorflow 1.11.0
    use method :
		下载bert的中文模型https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
		解压放到checkpoint的目录下即可
		├── checkpoint
		│   ├── chinese_L-12_H-768_A-12
		│   │   ├── bert_config.json
		│   │   ├── bert_model.ckpt.data-00000-of-00001
		│   │   ├── bert_model.ckpt.index
		│   │   ├── bert_model.ckpt.meta
		│   │   └── vocab.txt
		│   └── chinese_L-12_H-768_A-12.zip
		运行：
			python3 main.py（也可以根据代码设置命令行参数）
		代码详细使用说明见博客：https://www.jianshu.com/p/b05e50f682dd
			
	
