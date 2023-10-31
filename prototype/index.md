# Prototype设计模式


## 请举例说明克隆模式的其他应用。

### 应用描述

使用克隆模式实现深克隆：

在图表对象（DataChart）中包含一个数据集对象(DataSet)。数据集对象用于封装要显示的数据，用户可以通过界面上的复制按钮将该图表复制一份，复制后即可得到新的图表对象，然后可以修改新图表的编号、颜色、数据。

### UML类图

在该设计方案中，DataChart 类包含一个 DataSet 对象，在复制 DataChart 对象的同时将
复制 DataSet 对象，因此需要使用深克隆技术，可使用流来实现深克隆。其中Serializable是java.io包中定义的、用于实现Java类的序列化操作而提供的一个语义级别的接口。Serializable序列化接口没有任何方法或者字段，只是用于标识可序列化的语义。实现了Serializable接口的类可以被ObjectOutputStream转换为字节流，同时也可以通过ObjectInputStream再将其解析为对象。

![Figure 1-1 深克隆类图](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%E6%B7%B1%E5%85%8B%E9%9A%86%E7%B1%BB%E5%9B%BE_gFzoP2hb3B.png)

### 代码实现

#### DataSet

```java
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class DataSet implements Serializable {
    private List data = new ArrayList<Integer>();

    public List getData() {
        return data;
    }
    public void setData(List data) {
        this.data = data;
    }
    public void addData(int value) {
        data.add(value);
    }
    public Integer getData(int pos) {
        return (Integer) data.get(pos);
    }
    public void removeData(int pos) {
        data.remove(pos);
    }
    public int getLength() {
        return data.size();
    }
}
```

#### DataChart

```java
import java.io.*;

public class DataChart implements Serializable {
    /**
     * ds : 图表数据集
     * color : 图表颜色
     * no ： 图表编号
     */
    private DataSet ds = new DataSet();
    private String color;
    private int no;

    public DataSet getDs() {
        return ds;
    }
    public void setDs(DataSet ds) {
        this.ds = ds;
    }
    public String getColor() {
        return color;
    }
    public void setColor(String color) {
        this.color = color;
    }
    public int getNo() {
        return no;
    }
    public void setNo(int no) {
        this.no = no;
    }

    /**
     * 打印图表
     */
    public void display() {
        System.out.println("图表编号：" + this.getNo());
        System.out.println("图表颜色：" + this.getColor());
        System.out.println("图表数据集");
        for (int i = 0; i < ds.getLength(); ++ i ) {
            System.out.println(i + " : " + ds.getData(i));
        }
    }

    /**
     * 使用序列化技术实现深克隆
     */
    public DataChart deepClone() throws IOException, ClassNotFoundException, OptionalDataException {
        // 将对象写入流中
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(byteArrayOutputStream);
        objectOutputStream.writeObject(this);

        // 将对象从流中取出
        ByteArrayInputStream byteArrayInputStream = new ByteArrayInputStream(byteArrayOutputStream.toByteArray());
        ObjectInputStream objectInputStream = new ObjectInputStream(byteArrayInputStream);
        return (DataChart) objectInputStream.readObject();
    }
}
```

#### 测试代码

```java
import java.util.ArrayList;
import java.util.List;

public class Client {

    public static void main(String[] args) {
        DataChart dataChart1 = null, dataChart2 = null;
        dataChart1 = new DataChart();
        dataChart1.setColor("red");
        dataChart1.setNo(1);
        List data = new ArrayList<Integer>();
        data.add(1);
        data.add(2);
        DataSet ds = new DataSet();
        ds.setData(data);
        dataChart1.setDs(ds);
        try {
            // 调用深克隆方法创建一个克隆对象
            dataChart2 = dataChart1.deepClone();
            System.out.println(dataChart1 == dataChart2);
            System.out.println(dataChart1.getDs() == dataChart2.getDs());
            System.out.println(dataChart1.getNo() == dataChart2.getNo());
            System.out.println(dataChart1.getColor() == dataChart2.getColor());
        } catch (Exception e) {
            System.out.println("克隆失败！");
        }
    }
}
```

#### 测试结果

该结果符合深克隆的引用对象也被复制的现象，因为两个引用对象不相等。

![Figure 1-2](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_pVVRZiF0cc.png)

## 试描述浅克隆和深克隆。

根据在复制原型对象的同时是否复制包含在原型对象中引用类型的成员变量，原型模式的克隆机制可分为两种，即浅克隆（Shallow Clone）和深克隆（Deep Clone）。

### 浅克隆

在浅克隆中，如果原型对象的成员变量是值类型，将复制一份给克隆对象；如果原型对象的成员变量是引用类型，则将引用对象的地址复制一份给克隆对象，也就是说原型对象和克隆对象的成员变量指向相同的内存地址。简单来说，在浅克隆中，当对象被复制时只复制它本身和其中包含的值类型的成员变量，而引用类型的成员对象并没有复制。

![Figure 2-1 浅克隆](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_KsRGdSrmyN.png)

### 深克隆

在深克隆中，无论原型对象的成员变量是值类型还是引用类型，都将复制一份给克隆对象，深克隆将原型对象的所有引用对象也复制一份给克隆对象。简单来说，在深克隆中，除了对象本身被复制外，对象所包含的所有成员变量也将复制。

![Figure 2-2 深克隆](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_LoOArQb3Nh.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/prototype/  

