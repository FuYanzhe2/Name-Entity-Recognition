# Name-Entity-Recognition
Lstm-crf,Lattice-CRF,bert-ner及近年ner相关论文follow<br>

- ChineseNER 中文NER <br>
	tensorflow 1.4.0<br><br>
    use method :<br>
        python3 main.py<br>
    
    详细使用原来即实验结果见博客https://www.jianshu.com/p/aed50c1b2930<br>

- fyz_lattice_NER 中文NER lattice model<br>
	pytorch 0.4.0<br>
	Python 3.6<br>
	use method :<br>
        python3 main.py<br>
		或者直接配置然后运行：bash fyz_run_decode.sh<br>
	
	详细使用原来即实验结果见[博客](https://www.jianshu.com/p/9c99796ff8d9)<br>
	文件中需要的两个[词向量地址](https://pan.baidu.com/s/1Uj97799tGjdET_vbdkW7tQ )：<br>
		提取码：vgwi <br>
		解压文件 放到data/ 文件夹下即可<br>
		
- BERT-BiLSTM-CRF-NER<br>
	tensorflow 1.11.0<br>
    use method :<br>
		下载bert的[中文模型](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)<br>
		解压放到checkpoint的目录下即可<br>
		运行：<br>
			python3 main.py（也可以根据代码设置命令行参数）<br>
		代码详细使用说明见[博客]：(https://www.jianshu.com/p/b05e50f682dd)<br>
			
	
