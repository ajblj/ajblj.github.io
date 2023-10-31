# Builder设计模式


## 修改本例，增加一个新的concrete的Builder。

### Builder 模式结构

建造者模式的UML类图如下：

![Figure 1-1 Builder类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E7%B1%BB%E5%9B%BE_BDgVJkv0EU.png)

由上图可知，建造者模式包含以下四个角色：

-   Builder（抽象建造者）：它为创建一个产品对象的各个部件指定抽象接口，在该接口中一般声明两类方法，一类方法是buildPartX()，它们用于创建复杂对象的各个部件；另一类方法是getResult()，它们用于返回复杂对象。Builder既可以是抽象类，也可以是接口。
-   ConcreteBuilder（具体建造者）：它实现了Builder接口，实现各个部件的具体构造和装配方法，定义并明确所创建的复杂对象，还可以提供一个方法返回创建好的复杂产品对象（该方法也可以由抽象建造者实现）。
-   Product（产品）：它是被构建的复杂对象，包含多个组成部件，具体建造者创建该产品的内部表示并定义它的装配过程。
-   Director（指挥者）：指挥者又称为导演类，它负责安排复杂对象的建造次序，指挥者与抽象建造者之间存在关联关系，可以在其construct()建造方法中调用建造者对象的部件构造与装配方法，完成复杂对象的建造。

### Builder 模式实现

Builder模式的典型代码如下：

-   Product

```java
public class Product {
    private String partA; // 定义部件，部件可以是值类型和引用类型
    private String partB;
    private String partC;
    // 属性的Getter和Setter方法省略
}

```

-   Builder

```java
public abstract class Builder {
    // 创建产品对象
    protected Product product = new Product();
    
    public abstract void buildPartA();
    public abstract void buildPartB();
    public abstract void buildPartC();
    
    // 返回产品对象
    public Product getResult() {
        return product;
    }
}

```

-   ConcreteBuilder

```java
public class ConcreteBuilder1 extends Builder {
    public void buildPartA() {
        product.setPartA("A1");
    }
    public void buildPartB() {
        product.setPartA("B1");
    }
    public void buildPartC() {
        product.setPartA("C1");
    }
}

```

-   Director

```java
public class Director {
    private Builder builder;
    
    public Director(Builder builder) {
        this.builder = builder;
    }
    public void setBuilder(Builder builder) {
        this.builder = builder;
    }
    // 产品的构建与组装方法
    public Product construct() {
        builder.buildPartA();
        builder.buildPartB();
        builder.buildPartC();
        return builder.getResult();
    }
}

```

### 增加一个新的ConcreteBuilder

#### JsonBuilder类

增加一个JsonBuilder的具体建造者，它可以产生一个json格式的文件，其中title设置对象的名称，str设置列表的名称，items设置列表中的内容。

```java
public class JsonBuilder extends Builder {
    private StringBuffer buffer = new StringBuffer();  // 开始在此属性建立文件
    public void makeTitle(String title) {    // 一般文字格式的标题
        buffer.append("\"" + title + "\":\n{\n"); //对象的名称
    }
    public void makeString(String str) {    // 一般文字格式的字串
        buffer.append("\t\"" + str + "\": ["); //列表的名称
    }
    public void makeItems(String[] items) {    // 一般文字格式的项目
        for (int i = 0; i < items.length - 1; i++) {//列表内容
            buffer.append("\"" + items[i] + "\", ");
        }
        buffer.append("\"" + items[items.length - 1] + "\"");
        buffer.append("]\n");        // 空行
    }
    public Object getResult() {        // 完成的文件
        buffer.append("}\n");
        return buffer.toString();    // 把StringBuffer转换成String
    }
}

```

#### 测试类

修改Main测试类，增加Json选项：

![Figure 1-2](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_eD2-GDozDZ.png)

运行结果：

其中Title为"Creeting"在json中被设置为对象名，String为"从早上到白天结束"在json中被设置为列表名，Items为"早安。","午安。"在json中被设置为列表内容。

![Figure 1-3](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_lDFPZ7oD3V.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/builder/  

