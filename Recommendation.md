# 数据集

[Amozon etc.](https://cseweb.ucsd.edu/~jmcauley/datasets.html)

[Yelp](https://www.yelp.com/dataset/challenge)


---

1. [Interpretable Convolutional Neural Networks with Dual Local and Global Attention for Review Rating Prediction](http://delivery.acm.org/10.1145/3110000/3109890/p297-seo.pdf?ip=202.113.176.124&id=3109890&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EE4E04C281054793F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1557369301_04a67f3c6955770e18b74bdd0a515478)

本文对user、item依然是独立建模，采用独立的两个Attention建模Review文本信息从而获得Users/Items的向量表示。如下图所示

![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/AB4D5023BEF14EF2954365A57F04546F/7542)

局部Attention（L-Attn）和全局Attention（G-Attn）输入都是Review文本，不同之处在于L-Attn提取的是Word-Level的特征/突出的Word（滑动窗口size为5），G-Attn提取的是N-gram（2，3，4）级别的特征，会过滤掉不重要的词。Max-pooling过程会筛选出最突出的词/N-gram。


---
2. [Multi-Pointer Co-Attention Networks for Recommendation](https://arxiv.org/pdf/1801.09251.pdf)

本文学习User，Item的向量表示过程不独立，层次Attention的方式，选择Reveiw作为下一层Word-level Attention的输入。

![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/77E5D109BBE348459317DEE4F2CE881D/7574)

模型架构方面：User Review和Item Review 矩阵求的Affinity Matrix，然后分别按照行和列做Max-Pooling分别得到原始review的权重。soft-max本文采用的是Gumbel-Max。


---
3. [Why I Like it: Multi-task Learning for Recommendation and Explanation](http://delivery.acm.org/10.1145/3250000/3240365/p4-lu.pdf?ip=202.113.176.6&id=3240365&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EE4E04C281054793F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1557151348_f3dd38eb6cc6e8b3b704e113c22b1b40)

本文采用多任务学习框架，使得评分预测和评论生成相互提升。简单来说，采用矩阵分解做评分预测，并用sequence-to-sequence的模型做explanation生成，也就是对给定的（recommendation，user）生成个性化评论。

其他也有工作做E推荐模型的xplanation或模型的解释性，本文通过Generate review的方式做解释模型。其他的方式还包括：选出usefulness的review，预测用户可能对推荐item的opinionated content（为什么用户会关注推荐的那些features）



- Related-Work部分,分别介绍了
    - 传统CF
    - 解决稀疏问题和透明度问题（从user－generated reviews中挖掘opinions和features）
    - sequence-to-sequence深度神经网络模型习
    - 基于GAN的文本生成这几个方面的相关研究
    - 以及本文与其他工作的区别。

![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/C462DE0F3E764644A60B241AE6609AED/7876)

- 模型
    - review于用户，反应用户的偏好；于item，反应item的特征。因而，采用同一种框架，两组参数分别处理用户的评论和对item的评论，以此捕获这种不同。
    - 在文本生成上采用对抗式训练，生成器采用Policy Gradient（基于GRU的seq2seq，包含Encoder（反应了用户偏好）, Decoder），判别器用CNN。
    - 使用强化学习的评论生成对抗训练


[blog](https://blog.csdn.net/lthirdonel/article/details/88595773)


- 优化算法：
    - 对抗训练模块和rating预测模块交替训练
    - 对抗训练模块内部交替训练：
        - teacher－forcing和Policy Gradient交替更新生成器参数
        - 生成器参数和判别器参数交替更新

- 实验
    - Item推荐
    - Explanation质量: 生成文本质量
        - Perplexity
        - tf-idf
    - 判别器Performance


---
4. 2019 WSDM [Review-Driven Answer Generation for Product-Related Question in E-commerce](http://delivery.acm.org/10.1145/3300000/3290971/p411-chen.pdf?ip=202.113.176.112&id=3290971&acc=ACTIVE%20SERVICE&key=BF85BBA5741FDC6E%2EE4E04C281054793F%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1557578087_efab547b617c9009cd33a70563d448d5)

- 问题:
    - 电商交易中，在购买过用户的评论中找出拟购买用户提出的产品特性相关问题的答案；
- 挑战：
    - 购买用户的评论中存在语法不正确，标点使用不规范等状况；
    - 对拟购买用户问题的回答文本不应该包含很多不相关信息
- 方法
    - 从商品相关的评论中抽取与问题相关的snippets
    - 从包含噪音的评论snippets中抽取相关的信息，并用它指导答案的生成

模型框架：
![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/A86B4874210B462197624144EA3E377A/7706)

在文本生成方面采用的是 Convolutional Sequence-to-Sequence。
- Encoder
    - 输入Embedding中包含，词embedding，POS tag embedding（包含了句子的句法结构和每个词的语法角色），绝对位置embedding
    - 激活函数采用Gated linear units(GLU)
- Generator
    - 基础的生成器，与Convolutional Sequence-to-Sequence一致，包含残差连接等方法。
    - 重点在于添加review-driven的导引，本文的做法也很常规
        - 在G卷积层的隐状态中增加一项，review snippet中词的加权和。
        - 用词在review snippet中出现的频率和与之相关性计算词的importance。

![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/A17D3EDB683D409282138E96BC7FC351/7704)

在数据处理上，构建auxiliary review snippet集合时采用WMD(Word Mover's Distance)作为衡量短文本相似度的指标

---
5. 2018 ICDM [A Reinfoecement Learning Framework for Explainable Recommendation](https://www.microsoft.com/en-us/research/uploads/prod/2018/08/main.pdf)

可解释性推荐基本包含两个部分：推荐模型和解释生成模型。传统的方法包含两种：后处理（侧重解释文本的质量）和嵌入式（侧重系统可解释性）。本文采用Wrapper的方式，利用强化学习将推荐模型和解释生成模型平行的处理。

强化学习构建：
- Environment：用户、商品、推荐模型、评测解释文本的背景知识
- Agent：包含两个Agent
    - Agent1：给定的用户、商品对，生成解释文本
    - Agent2：给定的用户、商品对，再加Agent1生成的文本，输出Rating Prediction，确保生成的解释可以预测用户的偏好
- State：用户和商品的特征向量表示
- Action：explanation和predicted rating
- Reward：反馈来源于两方面
    - 推荐系统（预测用户和商品的ratings）
    - 生成解释的质量（可读性、连贯性等）



---
6. 2018 CIKM[Multi-Source Pointer Network for Product Title Summarization](https://arxiv.org/pdf/1808.06885.pdf)

本文提出的这个任务场景比较独特，针对电商环境中商品标题summary任务。短标题更适用于移动端（智能手机）的屏幕显示，因而很有现实的意义。因为首次提出这个任务，因而作者基于淘宝平台构建了一个数据集，<O, K, S>，O表示产品原始标题，K表示产品的背景知识（品牌、产品名称等），S表示人工写的短标题。


- 模型框架
![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/BBD0A8FCCB4344E1BF81EB4D3449C0D2/8059)


在摘要中作者也提出过产品title摘要与传统的文本摘要任务不同的是：
- 不允许事实细节错误
- 不允许关键信息缺失

针对以上两点作者采用以下两种方法：
- 不产生不相关信息
- 保留关键信息（品牌名称、商品名称）


模型方面并没有提出改进摘要生成的方法，文中把Pointer-Generator应用在多个输入（源title，背景知识）上，得到多组attention权重和上下文向量。在解码阶段通过加权将两种attention权重组合起来，得到输出序列。

从摘要的角度看，本文提出的是一种抽取模型，而且融合了background knowledge（针对性保留原始title中的key information）。 

- 实验结果
![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/BE476D3803DD493AB460BC1CDCA3D146/8062)



----
7. 2018 ACL [Personalized Review Generation by Expanding Phrases and Attending on Aspect-Aware Representations](https://www.aclweb.org/anthology/P18-2112)

[==code==])(https://github.com/nijianmo/textExpansion)

本文做的是个性化评论生成，模型性能的提升得益于融合多方面的输入：
- Phrases序列：用Summary字段作为输入phrases；
- Attribute信息：
    - 参考[Learning to Generate Product Reviews from Attributes](https://www.aclweb.org/anthology/E17-1059)
- Aspect信息
    - 参考[An Unsupervised Neural Attention Model for Aspect Extraction](https://blog.csdn.net/u013695457/article/details/80390569)

![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/B2365AF1276D42DBA3FB335CC5442179/8137)


多输入通过attention融合起来，计算预测t时刻词的输出概率时Aspect的权重作为偏置，其他几个权重通过非线性融合。


----

8. 2017 ACL [Opinion Recommendation Using A Neural Network](https://www.aclweb.org/anthology/D17-1170)

- 任务
    - 给定其他用户对该商品的评价和该用户对其他商品的评价，opinion推荐要生成指定rating 得分/review。
- 动机
    - 通用的score是所有用户的平均水平，并不一定与个体用户的taste吻合；
    - 每个item可能存在成千上百的评论，用户不可能每个都看，因而需要一个简短的摘要
- 模型
    - 用户模型
    - 邻居模型
        - 邻居：和指定用户有相似taste的用户；
        - 关键点：找到指定用户的邻居。本文采用矩阵分解的方法，近似用户产品矩阵；
    - 商品模型
        - 动态Memory Network比Attention能更好的捕获抽象语义信息；
        - Multiple hops：每一层都是Attention + 线性转换
        - 输出Customized Product Representation V_C
    - 用户评论生成
        - LSTM 作为Decoder实现文本生成
    - 用户Rating预测


![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/0282CFCFCBEF426C8DE3A13B7FE02822/8198)

- 实验
    - 数据：Yelp
    - 对比实验
        - review预测： ROUGE-1.5.5
        - rating得分预测：MSE（Mean Square Error）
        
----
9. [Neural Attentional Rating Regression with Review-level Explanations](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf)

论文的思路很简单，通过attention的方式探索review的usefulness。实验充分，印证了前面提出的观点。


- 动机
    - 一般的方法使用review文本增强推荐效果，并没有考虑review文本的usefulness。本文通过Attention机制探索review文本的usefulness，从而为用户更好的做决策提供高质量的review。
    - usefulness定义：是否包含item的详细信息并且能帮助users更容易地作出购买决定。
    - 以往的review重要处理方式：Latest和Top_Rated_Useful，都有缺陷
- 方法
    - Latent Factor Model
    - CNN 
    
![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/EAE8DA825BA14BAA87763EAB1EDBB3F1/8466)

    - 组合预测模型
![](https://note.youdao.com/yws/public/resource/5d662639c1f5c75972bd342046b4ed4f/xmlnote/A1299119FE21426FAA5BED6DBFBA7616/8469)
    
- 实验
    - Rating Prediction实验
    - 参数分析
    - case分析
    - 可解释性研究
        - Review－level Explanation
        - Usefulness in Terms of User Rated
        - Crowd-sourcing based Usefulness Evaluation (只对比Top_Rated_Useful选出的Review)
            - Review-level Usefulness Analysis
            - Pairwise Usefulness Analysis

