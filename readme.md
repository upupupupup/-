以图搜图
思路：
1.输入
2.提取图片特征 img_feat
3.img_feat和已存储的图片特征db_feat进行对比
    -范式  --> 距离排序
    -余弦相似度    
    
问题：
1.使用训练好的模型进行特征提取
    -训练模型
    -测试
2.提取已有图片的特征并存储
    -名称+特征 db_name,db_feat
3.对比
4.排序，最相似

代码：
1.初始化
    -加载网络
        -加载参数
        -开启验证模式model.eval()
2.初始化已存储图片的特征
    - init img_feat
3.加载特征
    - load img_feat
4.对比
    cal_similarity