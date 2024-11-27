设 $$z_i$$和$$z_j$$分别为第$$i$$个和第$$j$$个样本的特征表示，$$T$$为温度参数。对于一个给定的样本对$$(z_i, z_j)$$，其中 $$z_i$$和 $$z_j$$是正样本（即它们来自同一图像的不同变换），损失函数$$L$$可以表示为：
$$
L = -\frac{1}{2N} \sum_{i=1}^{N} \sum_{\substack{j=1 \\ j \neq i}}^{2N} \log \frac{\exp\left(\frac{z_i \cdot z_j}{T}\right)}{\sum_{\substack{k=1 \\ k \neq i}}^{2N} \exp\left(\frac{z_i \cdot z_k}{T}\right)}
$$
其中$$N$$是批次大小，$$2N$$是因为$$out1$$和$$out2$$被拼接在一起，所以总的样本数是原来的两倍。这个公式包含了两个部分：

1. 分子：$$\exp\left(\frac{z_i \cdot z_j}{T}\right)$$，这是正样本对之间的相似度，通过点积和温度参数$$T$$缩放后，再应用指数函数
2. 分母：$$\sum_{\substack{k=1 \\ k \neq i}}^{2N} \exp\left(\frac{z_i \cdot z_k}{T}\right)$$，这是样本$$z_i$$与所有其他样本（包括正样本和负样本）之间的相似度之和，用于归一化分子，使得损失函数具有概率分布的性质

