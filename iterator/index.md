# Iterator设计模式


## 针对Iterator的例子，将存储Book用的数组换成其他Collection并运行。

将数组存储方式换成ArrayList存储。

### 类图

![img](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/%25E7%25B1%25BB%25E5%259B%25BE.png)

### Aggregate接口

聚合定义创建相应迭代器对象的接口

```java
public interface Aggregate{
    public abstract Iterator iterator();
}
```

### Iterator接口

迭代器定义访问和遍历元素的接口

```java
public interface Iterator {
    public abstract boolean hasNext();
    public abstract Object next();
}

```

### Book类

定义元素

```java
public class Book {
    private String name;
    public Book(String name){
        this.name = name;
    }

    public String getName(){
        return name;
    }
}

```

### BookShelf类

具体聚合实现创建相应迭代器的接口，该操作返回BookShelfIterator的一个适当的实例，这里将存储Book的Collection换成了ArrayList。

```java
import java.util.ArrayList;
import java.util.List;

public class BookShelf implements Aggregate{
    private List<Book> books;

    public BookShelf(Integer maxSize){
        books = new ArrayList<Book>(maxSize);
    }

    public Book getBookAt(Integer idx){
        return books.get(idx);
    }

    public void appendBook(Book book){
        books.add(book);
    }

    public Integer getLength(){
        return books.size();
    }

    @Override
    public Iterator iterator() {
        return new BookShelfIterator(this);
    }
}

```

### BookShelfIterator 类

-   具体迭代器实现迭代器接口
-   对该聚合遍历时跟踪当前位置

```java
public class BookShelfIterator implements Iterator{
    private BookShelf bookShelf;
    private Integer index;

    public BookShelfIterator(BookShelf bookShelf){
        this.bookShelf = bookShelf;
        this.index = 0;
    }

    @Override
    public boolean hasNext() {
        if(index >= bookShelf.getLength()) return false;
        else return true;
    }

    @Override
    public Object next() {
        Book book = bookShelf.getBookAt(index ++);
        return book;
    }
    
}

```

### Main类

```java
public class Main {
    public static void main(String[] args) {
        BookShelf bookShelf = new BookShelf(4);
        bookShelf.appendBook(new Book("Around the World in 80 Days"));
        bookShelf.appendBook(new Book("Bible"));
        bookShelf.appendBook(new Book("Forrest Gump"));
        bookShelf.appendBook(new Book("Triumph"));

        Iterator it = bookShelf.iterator();
        while(it.hasNext()){
            Book book = (Book) it.next();
            System.out.println(book.getName());
        }
    }
}

```

### 运行结果

运行Main类测试代码后，结果如下：

![Figure 1-1](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_med6ptwPKo.png)

## 针对Iterator的例子，设计一个Specified的Iterator并运行。

### 设计实现

设计实现一个由后向前遍历和获取指定下标的Iterator，重写Iterator接口中的方法，并实现该接口：

ReverseIterator接口：

```java
public interface ReverseIterator {
    //检查有无前一个元素
    public abstract boolean hasPre();
    //返回聚合当中的一个元素
    public abstract Object getPre();
    //返回聚合中的特定元素
    public abstract Object getByIndex(int index);
}

```

BookShelfReverseIterator实现类：

```java
public class BookShelfReverseIterator implements ReverseIterator {

    private BookShelf bookShelf;
    private int index;
    
    public BookShelfReverseIterator(BookShelf bookShelf) {
        this.bookShelf = bookShelf;
        this.index = bookShelf.getLength() - 1;
    }
    public boolean hasPre() {
        if(index >= 0) {
            return true;
        }
        return false;
    }

    public Object getPre() {
        Book book = bookShelf.getBookAt(index);
        index--;
        return book;
    }
    
    public Object getByIndex(int index) {
        if(index >= 0 && index < bookShelf.getLength()) {
            Book book = bookShelf.getBookAt(index);
            return book;
        }
        return null;
    }
}

```

测试代码如下所示：

```java
public class ReverseMain {
    public static void main(String[] args) {
        BookShelf bookShelf = new BookShelf(4);
        bookShelf.appendBook(new Book("Around the World in 80 Days"));
        bookShelf.appendBook(new Book("Bible"));
        bookShelf.appendBook(new Book("Forrest Gump"));
        bookShelf.appendBook(new Book("Triumph"));
        
        ReverseIterator rit = bookShelf.rIterator();

        System.out.println("由后向前遍历");
        while(rit.hasPre()){
            Book book = (Book) rit.getPre();
            System.out.println(book.getName());
        }

        System.out.println("获取指定元素");
        Book book = (Book) rit.getByIndex(2);
        System.out.println(book.getName());
    }
}

```

### 运行结果

![Figure 2-1](https://cdn.jsdelivr.net/gh/ajblj/blogImage@main/d2l/image_Dbu6X3fL2t.png)

---

> 作者: [jblj](https://github.com/ajblj/)  
> URL: http://example.org/iterator/  

