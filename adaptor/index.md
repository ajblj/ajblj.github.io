# Adaptor设计模式


## 什么是双向适配器？

在对象适配器使用过程中，如果在适配器中同时包含对目标类和适配者类的引用，适配者可以通过它调用目标类的方法，目标类也可以通过它调用适配者类的方法，那么该适配器就是一个双向适配器；即双向适配器类可以把适配者接口转换成目标接口，也可以把目标接口转换成适配者接口。

比如，单向适配器只能把交流220V的电压转换为直流12V的电压，而双向适配器可以互相转换，即也可以把直流12V的电压转换为交流220V的电压。

## 举例说明实现方式

交流220V电压和直流12V电压的相互转换：

交流220V：接口+实现类

直流12V：接口+实现类

转换器：适配器类

交流220V接口和实现类：

![Figure 2-1](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image.png)

![Figure 2-2](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_1.png)

直流12V接口和实现类：

![Figure 2-3](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_2.png)

![Figure 2-4](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_3.png)

转换器的实现：

![Figure 2-5](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_4.png)

测试：

![Figure 2-6](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_5.png)

测试结果：

![Figure 2-7](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_6.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/adaptor/  

