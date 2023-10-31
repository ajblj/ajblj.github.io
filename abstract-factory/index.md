# Abstract Factory设计模式


## 阅读Abstract Factory的例子的代码，举例说明使用Abstract Factory模式的其他应用。

### 抽象工厂模式结构

![Figure 1-1 抽象工厂模式类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E6%8A%BD%E8%B1%A1%E5%B7%A5%E5%8E%82%E6%A8%A1%E5%BC%8F%E7%B1%BB%E5%9B%BE_1X565k_xM1.png)

由上图可知，抽象工厂模式包含以下四个角色：

-   AbstractProduct（抽象产品）：它为每种产品声明接口，在抽象产品中声明了产品所具有的业务方法。
-   Product1（具体产品）：它定义具体工厂生产的具体产品的具体产品对象，实现抽象产品接口中声明的业务方法。
-   AbstractFactory（抽象工厂）：它声明了一组用于创建一族产品的方法，每一个方法对应一种产品。
-   ConcreteFactory1（具体工厂）：它实现了在抽象工厂中声明的创建产品的方法，生成一组具体产品，这些产品构成了一个产品族，每一个产品都位于某个产品等级结构中。

### 抽象工厂模式实现

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

### 抽象工厂模式应用

![Figure 1-2 抽象工厂](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E6%8A%BD%E8%B1%A1%E5%B7%A5%E5%8E%82_9F29euj-NE.png)

其中，接口AbstractFactory充当抽象工厂，其子类WindowsFactory、UnixFactory和LinuxFactory充当具体工厂；Text和Button充当抽象产品，其子类WindowsText、UnixText、LinuxText和WindowsButton、UnixButton、LinuxButton充当具体产品。

#### 抽象工厂及其具体工厂

-   AbstractFactory

```java
public abstract class AbstractFactory {
    //创建Button对象
    public abstract Button createButton();

    //创建Text对象
    public abstract Text createText();
}
```

-   WindowsFactory

```java
public class WindowsFactory extends AbstractFactory {
    @Override
    public Button createButton() {
        return new WindowsButton();
    }
    @Override
    public Text createText() {
        return new WindowsText();
    }
}
```

-   UnixFactory

```java
public class UnixFactory extends AbstractFactory {
    @Override
    public Button createButton() {
        return new UnixButton();
    }
    @Override
    public Text createText() {
        return new UnixText();
    }
}
```

-   LinuxFactory

```java
public class LinuxFactory extends AbstractFactory {
    @Override
    public Button createButton() {
        return new LinuxButton();
    }
    @Override
    public Text createText() {
        return new LinuxText();
    }
}
```

#### Button抽象类及其具体类

-   Button抽象类

```java
public abstract class Button{
    //点击按钮事件
    public abstract void click();
}
```

-   WindowsButton

```java
public class WindowsButton extends Button{
    @Override
    public void click(){
        System.out.println("点击Windows按钮");
    }
}

```

-   UnixButton

```java
public class UnixButton extends Button{
    @Override
    public void click(){
        System.out.println("点击Unix按钮");
    }
}

```

-   LinuxButton

```java
public class LinuxButton extends Button{
    @Override
    public void click(){
        System.out.println("点击Linux按钮");
    }
}

```

#### 测试函数

测试代码如下：

```java
public class Main {
    public static void main(String[] args) {
        AbstractFactory factory = new WindowsFactory();
        Button button = factory.createButton();
        Text text = factory.createText();
        button.click();
        text.show();
    }
}

```

测试结果如下：

![Figure 1-3](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_XDoHchgVt_.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/abstract-factory/  

