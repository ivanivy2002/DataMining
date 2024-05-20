<center><h3>21302010040 叶天逸</h3></center>
<center>主机用户名: </center>
<center>Win usr@hostname: Lenovo@IVANIVY-SAVIOR</center>
<center>WSL usr@hostname: ivan@IVANIVY-SAVIOR</center>
<center>2024-05-20
20:17</center>

## 算法分析

### Kmeanspp
1. 初始化质心，使用小心选择，选择相距较远的点为初始质心
2. 计算距离，分配簇，计算每个簇的质心，重新分配
3. 迭代上一步
### EM
EM比较简单，先初始化质心，我使用随机方法，不然效果太好，直接就1.0了哈哈
1. E-step，计算权重
2. M-step，更新中心
3. 迭代上面两步，直至通过设定的门槛
## 结果
![[Pasted image 20240520192342.png|194]]
## 详细的探索和讨论
### Kmeanspp
在簇数量没有事先约定的情况下，我们希望确定簇的数量
反复测试K=5的情况：

![[Pasted image 20240520195016.png|158]]
这是怎么回事呢

使用肘部法来获取可能的最优簇数：

![[Pasted image 20240520195225.png|202]]
看到K=6可能是最优的

![[Pasted image 20240520195103.png|132]]
但是6没法达到最优解，

![[Pasted image 20240520200502.png|475]]
![[Pasted image 20240520200431.png|475]]

通过绘图我们能知道，应该有5个簇
所以是这样的，初始化仍然是随机概率，所以有概率选到不好的初始质心
但是显然1.0 是正确的解答
