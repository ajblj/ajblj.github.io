# Bridge设计模式


## 试用Bridge模式完成下列事情：饮料的杯子有大、中、小；行为有：加奶，加糖，啥都不加。

### Bridge 模式结构

桥机器模式的UML类图如下：

![Figure 1-1 桥接器类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/Main_RV7GXheGdz.png)

由上图可知，桥接器模式包含以下四个角色：

-   Abstraction（抽象类）：它是用于定义抽象类的接口，通常是抽象类而不是接口，其中定义了一个Implementor（实现类接口）类型的对象并可以维护该对象，它与Implementor之间具有关联关系，它既可以包含抽象业务方法，也可以包含具体业务方法。
-   RefinedAbstraction（扩充抽象类）：它扩充由Abstraction定义的接口，通常情况下它不再是抽象类而是具体类，实现了在Abstraction中声明的抽象业务方法，在RefinedAbstraction中可以调用在Implementor中定义的业务方法。
-   Implementor（实现类接口）：它是定义实现类的接口，这个接口不一定要与Abstraction的接口完全一致，事实上这两个接口可以完全不同。一般而言，Implementor接口仅提供基本操作，而Abstraction定义的接口可能会做更多更复杂的操作。Implementor接口对这些基本操作进行了声明，而具体实现交给其子类。通过关联关系，在Abstraction中不仅拥有自己的方法，还可以调用到Implementor中定义的方法，使用关联关系代替继承关系。
-   ConcreteImplementor（具体实现类）：它具体实现了Implementor接口，在不同的ConcreteImplementor中提供基本操作的不同实现，在程序运行时ConcreteImplementor对象将替换其父类对象，提供给抽象类具体的业务操作方法。

### Bridge 模式实现

Bridge模式的典型代码如下：

-   Implementor

```java
public interface Implementor {
    public void operationImpl();
}

```

-   ConcreteImplementor

```java
public class ConcreteImplementor implements Implementor {
    public void operationImpl() {
        //具体业务方法的实现
    }
}

```

-   Abstraction

```java
public abstract class Abstraction {
    protected Implementor impl; //定义实现类接口对象

    public void setImpl(Implementor impl) {
        this.impl = impl;
    }

    public abstract void operation(); //声明抽象业务方法
}

```

-   RefinedAbstraction

```java
public class RefinedAbstraction extends Abstraction {
    public void operation() {
        //业务代码
        impl.operationImpl(); //调用实现类的方法
        //业务代码
    }
}

```

### 修改Bridge模式的“咖啡”例子

#### 修改后的类图

![Figure 1-2 咖啡类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E5%92%96%E5%95%A1%E7%B1%BB%E5%9B%BE_CEAIJPb_Jn.png)

#### 修改杯子大小维度

增加小杯具体类，用重复次数来说明是冲中杯还是大杯还是小杯 ，重复1次是小杯。

```java
//小杯
public class SmallCoffee extends Coffee{
    public SmallCoffee() {
        setCoffeeImp();
    }
    public void pourCoffee(){
        CoffeeImp coffeeImp = this.getCoffeeImp();
        //我们以重复次数来说明是冲中杯还是大杯还是小杯 ,重复1次是小杯
        coffeeImp.pourCoffeeImp();
        System.out.println("小杯来了" );
    }
}

```

#### 修改添加物品维度

增加加糖具体类。

```java
//加糖
public class SugarCoffeeImp extends CoffeeImp{
    public void pourCoffeeImp(){
        System.out.println("加了甜甜的糖浆");
    }
}
```

#### 测试

测试代码：

使用CoffeeImpSingleton可以设定加什么物品，再用定义杯子大小的具体类进行咖啡的设置。

![Figure 1-3](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_AOo_h-AJCI.png)

测试结果：

![Figure 1-4](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_EbVjgcKMvM.png)





---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/bridge/  

