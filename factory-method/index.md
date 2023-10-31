# Factory Method设计模式


## 改写本例，用于添加另一个具体工厂和具体产品。

### 工厂方法模式结构

![Figure 1-1 工厂模式方法类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E6%96%B9%E6%B3%95%E7%B1%BB%E5%9B%BE_HI8K47rsPc.png)

### 具体类图

![Figure 1-2 类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E7%B1%BB%E5%9B%BE_c6dTCFLFdG.png)

### 抽象工厂和抽象产品

-   抽象工厂：Factory 是产生Product的抽象类，具体内容由ConcreteFactory决定。Factory对于实际产生的ConcreteProduct一无所知，唯一知道的就是调用Product和产生新对象的方法。该类描述的是框架。

```java
package framework;

public abstract class Factory {
    public final Product create(String owner) {
        Product p = createProduct(owner);
        registerProduct(p);
        return p;
    }

    protected abstract Product createProduct(String owner);
    protected abstract void registerProduct(Product Product);
}
```

-   抽象产品：规定了此Pattern所产生的对象实例应有的接口，具体内容则由子类ConcreteProduct参与者决定，该类描述的是框架。

```java
package framework;

public abstract class Product {
    public abstract void use();
}

```

### 具体工厂

实际处理内容的部分，这里是一个创建账户的工厂。

```java
package account;

import java.util.Vector;
import framework.*;

public class AccountFactory extends Factory{

    private Vector<String> names = new Vector<String>();

    @Override
    protected Product createProduct(String name) {
        return new Account(name);
    }

    @Override
    protected void registerProduct(Product product) {
        names.add(((Account)product).getName());
    }

    public Vector<String> getNames() {
        return names;
    }
    
}

```

### 具体产品

实际处理内容的部分，定义了工厂创建的账户。

```java
package account;

import framework.*;

public class Account extends Product {
    private String name;
    Account(String name){
        System.out.println("创建" + name + "的账户。");
        this.name = name;
    }

    public void use() {
        System.out.println("登录" + name + "的账户。");
    }

    public String getName(){
        return name;
    }
}
```

### 结果显示

-   测试代码

```java
import account.AccountFactory;
import framework.*;

public class Main {
    public static void main(String[] args) {
        Factory factory = new AccountFactory(); 
        Product p1 = factory.create("张三");
        Product p2 = factory.create("李四");
        Product p3 = factory.create("王五");
        p1.use();
        p2.use();
        p3.use();
    }
}

```

-   测试结果

![Figure 1-3](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_QzH2AhooZY.png)

## 请举例说明其他的工厂模式的应用。

### 简单工厂模式

#### 简单工厂模式结构

![Figure 2-1 简单工厂类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E7%AE%80%E5%8D%95%E5%B7%A5%E5%8E%82%E7%B1%BB%E5%9B%BE_zSeBgI6Ped.png)

其中根据上图可知，简单工厂模式包含以下3个角色:

-   Factory（工厂角色）：即工厂类，它是简单工厂模式的核心，负责实现创建所有产品实例的内部逻辑；其可以被外界直接调用，创建所需的产品对象。
-   AbstractProduct（抽象产品角色）：它是工厂类创建的所有对象的父类，封装了各种产品对象的共有方法。
-   Product（具体产品角色）：它是简单工厂模式的创建目标，所有被创建的对象都充当这个角色的某个具体类的实例。

#### 简单工厂模式实现

三种角色的典型代码如下：

-   Factory

```java
public class Factory {
    // 静态工厂方法
    public static AbstractProduct createProduct(String arg) {
        AbstractProduct product = null;
        if (arg.equalsIgnoreCase("1")) {
            product = new Product1();
        } else if (arg.equalsIgnoreCase("2")) {
            product = new Product2();
        }
        return product;
    }
}

```

-   AbstractProduct

```java
public abstract class AbstractProduct {
    // 所有产品类的公共业务方法
    public void methodSame() {
        // 公有方法的实现
    }
    // 声明抽象业务方法
    public abstract void methodDiff() {
        
    }
}

```

-   Product

```java
public class Product1 extends AbstractProduct {
    // 实现业务方法
    public void methodDiff() {
        // 业务方法的实现
    }
}

```

#### 简单工厂模式应用

![Figure 2-2 简单工厂应用](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E7%AE%80%E5%8D%95%E5%B7%A5%E5%8E%82_F5zsBiLBol.png)

-   具体产品Circle、Rectangle、Triangle

```java
public class Circle extends Shape {
    @Override
    public void draw() {
        System.out.println("绘制圆形");
    }
    @Override
    public void erase() {
        System.out.println("擦除圆形");
    }
}

public class Rectangle extends Shape {
    @Override
    public void draw() {
        System.out.println("绘制长方形");
    }
    @Override
    public void erase() {
        System.out.println("擦除长方形");
    }
}

public class Triangle extends Shape {
    @Override
    public void draw() {
        System.out.println("绘制三角形");
    }
    @Override
    public void erase() {
        System.out.println("擦除三角形");
    }
}

```

-   抽象产品Shape

```java
public abstract class Shape {
    /**
     * 绘制图形
     */
    public abstract void draw();

    /**
     * 擦除图形
     */
    public abstract void erase();
}
```

-   简单工厂ShapeFactory

```java
public class ShapeFactory {
    public static Shape createShape(String type){
        Shape shape = null;
        if (type.equalsIgnoreCase("Circle")) {
            shape = new Circle();
        } else if (type.equalsIgnoreCase("Rectangle")) {
            shape = new Rectangle();
        } else if (type.equalsIgnoreCase("Triangle")) {
            shape = new Triangle();
        }
        return shape;
    }
}
```

### 抽象工厂模式

#### 抽象工厂模式结构

![Figure 2-3 抽象工厂模式类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E6%8A%BD%E8%B1%A1%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E7%B1%BB%E5%9B%BE_HeohoZ020B.png)

由上图可知，抽象工厂模式包含以下四个角色：

-   AbstractProduct（抽象产品）：它为每种产品声明接口，在抽象产品中声明了产品所具有的业务方法。
-   Product1（具体产品）：它定义具体工厂生产的具体产品的具体产品对象，实现抽象产品接口中声明的业务方法。
-   AbstractFactory（抽象工厂）：它声明了一组用于创建一族产品的方法，每一个方法对应一种产品。
-   ConcreteFactory1（具体工厂）：它实现了在抽象工厂中声明的创建产品的方法，生成一组具体产品，这些产品构成了一个产品族，每一个产品都位于某个产品等级结构中。

#### 抽象工厂模式实现

抽象工厂的典型代码如下：

-   AbstractFactory

```java
public interface AbstractFactory {
    public AbstractProduct1 createProduct1(); // 工厂方法一
    public AbstractProduct2 createProduct2(); // 工厂方法二
}

```

-   ConcreteFactory1

```java
public class ConcreteFactory1 implements AbstractFactory {
    // 工厂方法一
    public AbstractProduct1 createProduct1() {
        return new Product1();
    }
    // 工厂方法二
    public AbstractProduct2 createProduct2() {
        return new Product2();
    }
}

```

#### 抽象工厂模式应用

![Figure 2-4 抽象工厂应用](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E6%8A%BD%E8%B1%A1%E5%B7%A5%E5%8E%82_P3BrCFJW_c.png)

其中，接口AbstractFactory充当抽象工厂，其子类WindowsFactory、UnixFactory和LinuxFactory充当具体工厂；Text和Button充当抽象产品，其子类WindowsText、UnixText、LinuxText和WindowsButton、UnixButton、LinuxButton充当具体产品。

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/factory-method/  

