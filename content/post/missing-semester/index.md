---
author : "wdl"
title : "[MIT] The Missing Semester of Your CS Education 学习笔记"
date : "2023-02-21"
description : "如何精通工具"
tags : [
    "自学课程"
]
categories : [
    "SelfStudy"
]
math: true
---

## 前言

最近在想办法给博客加内容，然后就把这么古早的笔记都翻出来了。当时的学习动机、情景已经记不清了，看了看内容不禁想对当年的自己说“怎么这都要记”。反正简单的内容已经熟能生巧了，有些工具一直都没用过，记了也是很快就忘。

Anyway，这些内容还是有些用处的，所以就放上来了。应该有些地方是参考了[原文](https://missing-semester-cn.github.io/)的，具体也记不清了。

## shell

### 简介

bash 最广泛的shell

cmd快捷键：`Ctrl+Alt+T`

shell prompt（提示符）: 用户名 机器名 当前路径

 `$` 符号表示您现在的身份不是 root 用户 `#`表示是root用户（也表示注释）

命令commands : 如输入date回车

shell 基于空格分割命令并进行解析，然后执行第一个单词代表的程序，并将后续的单词作为程序可以访问的参数

打印：`echo`

```
echo hello   
echo "hello world"   
echo hello\ world
```

转义字符`\`

字符`!` `$` `\` `` ` 即使被双引号（`"`）包裹也具有特殊的含义，可以用单引号（`'`）



### 路径

环境变量：启动shell时设置

路径变量：显示路径列表 echo $PATH 在这些路径里搜索程序

遍历路径找到程序，运行

程序路径：`which`

```
which echo
```

```
/usr/bin/echo
```

从文件系统的顶部开始

windows每个驱动器都有单独路径结构，如C:\ 

绝对路径：以`/`开头 相对路径：相对当前位置

显示目前文件路径：`pwd`

进入子目录：`cd 目录名`

```
cd /home
```

`.`当前目录 `..`父目录`~`主目录`-`上一个访问的目录

```
cd ..
cd ./home
cd -
```

查看当前目录下所有文件：`ls`

大多数的命令接受标记和选项（带有值的标记），它们以 `-` 开头，并可以改变程序的行为

指令使用帮助：`指令名 --help`

```
ls -l //显示更多文件信息
```

```
ls -l /home
drwxr-xr-x 1 missing  users  4096 Jun 15  2019 missing
```

第一个字符 `d` 表示 `missing` 是一个目录。然后接下来的九个字符，每三个字符构成一组（`rwx`），它们分别代表了文件所有者（`missing`），用户组（`users`） 以及其他所有人具有的权限。其中 `-` 表示该用户不具备相应的权限。从上面的信息来看，只有文件所有者可以修改（`w`）`missing` 文件夹 （例如，添加或删除文件夹中的文件）。为了进入某个文件夹，用户需要具备该文件夹以及其父文件夹的“搜索”权限（以“可执行”`x`表示）。为了列出它的包含的内容，用户必须对该文件夹具备读权限（`r`）。对于文件来说，权限的意义也是类似的。注意，`/bin` 目录下的程序在最后一组，即表示所有人的用户组中，均包含 `x` 权限，也就是说任何人都可以执行这些程序。



### 命令

- 创建新文件夹 ：`mkdir 文件夹名`

- 创建新文件（类似创建空txt，打开可编辑）`touch 文件名`

- 移动文件 `mv 文件名 目标目录`

- 重命名文件 `cp 文件名 文件名`

- 拷贝文件 `cp 文件名 目标目录`

- 删除文件 `rm 文件名`

- 删除文件夹 `rm -r 目录名`

- 删除空文件夹`rmkir`

- 查看用户手册`man 文件名`（替代品`tldr`）

- 清除终端，回到顶部`Ctrl+L`

- 用浏览器打开html文件`xdg-open 文件名`

- 改变模式`chmod 模式 文件名`


```
chmod ugo+x 文件名 //添加x权限
```



### 程序间连接

组合多个程序，文件在不同程序之间传输

流stream 输入流输出流

默认指向终端  重定向流改变输入输出指向

将输入重定向为文件内容`< 文件名`

将输出重定向为文件内容`> 文件名`

```
echo hello > hello.txt 
```

打印文件内容：`cat`

```
cat hello.txt
cat < hello.txt
```

以上的输出都是内容hello

```
cat < hello.txt > hello2.txt //复制文件内容
```

追加内容而不是覆盖：`>>` 追加内容会覆盖（且会换行？）

```
cat < hello.txt >> hello2.txt //复制文件内容
```

空一行`echo -e >> 文件名`

管道符（pipe）`|`

将左边程序的输出作为右边程序的输入

只想得到输出的最后n行`tail -nl`

```
ls -l / | tail -nl > ls.txt
```

把前面输出的最后n行输出到ls.txt

`xargs` 可以将管道或标准输入（stdin）数据转换成命令行参数

之所以能用到这个命令，关键是由于很多命令不支持|管道来传递参数。xargs 一般是和管道一起使用

```
somecommand |xargs -item  command
```



### root用户

root用户可以做任何事

以root用户权限执行一些操作 `sudo`

更新当前系统软件列表：`sudo apt-get update`

向 `sysfs` 文件写入内容必须作为根用户才能做。系统被挂载在 `/sys` 下，`sysfs` 文件则暴露了一些内核（kernel）参数。 因此不需要借助任何专用的工具，就可以轻松地在运行期间配置系统内核（仅限Linux）

```
sudo find -L /sys/class/backlight -maxdepth 2 -name '*brightness*'
```

```
/sys/class/backlight/thinkpad_screen/brightness
```

```
cd /sys/class/backlight/thinkpad_screen
sudo echo 3 > brightness
```

```
An error occurred while redirecting file 'brightness'
open: Permission denied
```

原因：`|`、`>`、和 `<` 是通过 shell 执行的，而不是被各个程序单独执行。 `echo` 等程序并不知道 `|` 的存在，它们只知道从自己的输入输出流中进行读写。shell不是根用户

`su`可以以root用户身份获取一个shell

```
sudo su
//输入密码
echo 3 > brightness
```

```
echo 3 | sudo tee brightness
```

`tee`将输入的内容同时写入文件和屏幕上



### shell脚本

变量赋值的语法`foo=bar`

访问变量中存储的数值 `$foo`

以`'`定义的字符串为原义字符串，其中的变量不会被转义，而 `"`定义的字符串会将变量值进行替换

支持`if`, `case`, `while` 和 `for`

支持函数

```
mcd () {
    mkdir -p "$1"
    cd "$1"
}
```

- `$0` - 脚本名
- `$1` 到 `$9` - 脚本的参数。 `$1` 是第一个参数，依此类推。
- `$@` - 所有参数
- `$#` - 参数个数
- `$?` - 前一个命令的返回值（退出码）（正确为0，错误非0）
- `$$` - 当前脚本的进程识别码（PID）
- `!!` - 完整的上一条命令，包括参数。常见应用：当你因为权限不足执行命令失败时，可以使用 `sudo !!`再尝试一次。
- `$_` - 上一条命令的最后一个参数

退出码可以搭配 `&&`和 `||`使用

```
false && echo "Will not be printed"
```



### 替换

以变量的形式获取一个命令的输出，通过命令替换实现`$()`

进程替换， `<( CMD )` 会执行 `CMD` 并将结果输出到一个临时文件中，并将 `<( CMD )` 替换成临时文件名。这在我们希望返回值通过文件而不是STDIN传递时很有用。

 `diff <(ls foo) <(ls bar)` 显示文件夹 `foo` 和 `bar` 中文件的区别

```
#!/bin/bash

echo "Starting program at $(date)" # date会被替换成日期和时间

echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
    grep foobar "$file" > /dev/null 2> /dev/null
    # 如果模式没有找到，则grep退出状态为 1
    # 我们将标准输出流和标准错误流重定向到Null，因为我们并不关心这些信息
    if [[ $? -ne 0 ]]; then  # -ne相当于!=
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done
```

使用`grep` 搜索字符串 `foobar`，如果没有找到，则将其作为注释追加到文件中

在bash中进行比较时，尽量使用双方括号 `[[ ]]` 而不是单方括号 `[ ]`，这样会降低犯错的几率



### 通配

通配基于文件扩展名展开表达式

当你想要利用通配符进行匹配时，你可以分别使用 `?` 和 `*` 来匹配一个或任意个字符。当你有一系列的指令，其中包含一段公共子串时，可以用花括号`{}`来自动展开这些命令

```
convert image.{png,jpg}
# 会展开为
convert image.png image.jpg
```

```
touch {foo,bar}/{a..h}
```

```
mv *{.py,.sh} folder
```



### shebang

利用shellcheck定位脚本中的错误

脚本并不一定只有用 bash 写才能在终端里调用。比如说一段 Python 脚本将输入的参数倒序输出：

```
#!/usr/local/bin/python
import sys
for arg in reversed(sys.argv[1:]):
    print(arg)
```

`#!`称为Shebang  展示运行一个脚本的程序的所在路径

`#!/bin/bash`使用Bash shell执行文件

在 `shebang` 行中使用 `env`命令，利用环境变量定位：`#!/usr/bin/env python`

- 函数只能与shell使用相同的语言，脚本可以使用任意语言
- 函数仅在定义时被加载，脚本会在每次被执行时加载。但每次修改函数定义，都要重新加载一次。
- 函数会在当前的shell环境中执行，脚本会在单独的进程中执行。因此，函数可以对环境变量进行更改，比如改变当前工作目录，脚本则不行。脚本需要使用 `export`将环境变量导出，并将值传递给环境变量。



### 查找文件

```
# 查找所有名称为src的文件夹
find . -name src -type d
# 查找所有文件夹路径中包含test的python文件
find . -path '*/test/*.py' -type f
# 查找前一天修改的所有文件
find . -mtime -1
# 查找所有大小在500k至10M的tar.gz文件
find . -size +500k -size -10M -name '*.tar.gz'
# 删除全部扩展名为.tmp 的文件
find . -name '*.tmp' -exec rm {} \;
# 查找全部的 PNG 文件并将其转换为 JPG
find . -name '*.png' -exec convert {} {}.jpg \;
```

替代品`fd`，以模式`PATTERN` 搜索的语法是 `fd PATTERN`

`locate`使用一个由 `updatedb`负责更新的数据库，则只能通过文件名搜索



### 查找代码

查找文件里符合条件的字符串或正则表达式：`grep 字符串 文件名`

常用搭配  

- `-i`：忽略大小写进行匹配
- `-n`：显示匹配行的行号
- `-r(R)`：递归查找子目录中的文件
- `-l`：只打印匹配的文件名
- `-c`：只打印匹配的行数
- `-C`：获取查找结果的上下文（前和后几行）
- `-v`：将对结果进行反选

替代品`rg`

```
# 查找所有使用了 requests 库的文件
rg -t py 'import requests'
# 查找所有没有写 shebang 的文件（包含隐藏文件）
rg -u --files-without-match "^#!"
# 查找所有的foo字符串，并打印其之后的5行
rg foo -A 5
# 打印匹配的统计信息（匹配的行和文件的数量）
rg --stats PATTERN
```



### 查找命令

搜索历史记录：`history`

 `history | grep find` 打印包含find子串的命令

使用 `Ctrl+R` 对命令历史记录进行回溯搜索。敲 `Ctrl+R` 后输入子串来进行匹配，查找历史命令行

使用方向键上或下

`fzf` 一个通用模糊查找工具



### 文件夹导航

`fasd`使用命令 `z` 帮助我们快速切换到最常访问的目录：`z cool`

概览目录结构：`tree`





## Vim

### 简介

Vim 是一个多模态编辑器

Vim 是可编程的

Vim 避免了使用鼠标



### 编辑模式

Vim 的设计以大多数时间都花在阅读、浏览和进行少量编辑改动为基础，因此它具有多种操作模式：

- **正常模式**：在文件中四处移动光标进行修改
- **插入模式**：插入文本
- **替换模式**：替换文本
- **可视化模式**（一般，行，块）：选中文本块
- **命令模式**：用于执行命令

不同的操作模式下，键盘敲击的含义也不同。比如，`x` 在插入模式会插入字母 `x`，在正常模式会删除当前光标所在的字母，在可视模式下则会删除选中文块。

在左下角显示当前的模式，Vim 启动时的默认模式是正常模式

按下 `<ESC>`从任何其他模式返回正常模式。在正常模式，键入 `i` 进入插入 模式，`R` 进入替换模式，`v` 进入可视（一般）模式，`V` 进入可视（行）模式，`<C-v>` （Ctrl-V, 有时也写作 `^V`）进入可视（块）模式，`:` 进入命令模式。



### 缓存与窗口

Vim 会维护一系列打开的文件，称为“缓存”。一个 Vim 会话包含一系列标签页，每个标签页包含一系列窗口（分隔面板）。每个窗口显示一个缓存。 缓存和窗口不是一一对应的关系，窗口只是视角。一个缓存可以在多个窗口打开，甚至在同一个标签页内的多个窗口打开。这个功能很好用，比如在查看同一个文件的不同部分的时候。

Vim 默认打开一个标签页，这个标签也包含一个窗口。

用 `:sp` / `:vsp` 来分割窗口



### 命令行

在正常模式下键入 `:` 进入命令行模式

- `:q` 退出（关闭窗口）
- `:w` 保存（写）
- `:wq` 保存然后退出
- `:e 文件名` 打开要编辑的文件
- `:ls` 显示打开的缓存
- `:help 标题`打开帮助文档

```
:help :w
```



### 移动

- 基本移动: `hjkl` （左， 下， 上， 右）
- 词： `w` （下一个词）， `b` （词初）， `e` （词尾）
- 行： `0` （行初）， `^` （第一个非空格字符）， `$` （行尾）
- 屏幕： `H` （屏幕首行）， `M` （屏幕中间）， `L` （屏幕底部）
- 翻页： `Ctrl-u` （上翻）， `Ctrl-d` （下翻）
- 文件： `gg` （文件头）， `G` （文件尾）
- 行数： `:{行数}<CR>` 或者 `{行数}G` 
- 杂项： `%` （找到配对，比如括号或者 /* */ 之类的注释对）
- 查找：`f{字符}`， `t{字符}`， `F{字符}`， `T{字符}`查找/到 向前/向后 在本行的{字符}，`,` /`;` 用于导航匹配
- 搜索: `/{正则表达式}`，`n` / `N` 用于导航匹配



### 编辑

- `O` / `o` 在之上/之下插入行
- `d{移动命令}`删除 
  - 例如，`dw` 删除词, `d$` 删除到行尾, `d0` 删除到行头,`dd`删除一整行。
- `c{移动命令}`改变
  - 例如，`cw` 改变词
  - 等效于 `d{移动命令}` 再 `i`
- `x` 删除字符（等同于 `dl`）
- `s` 替换字符（等同于 `xi`）
- 可视化模式 + 操作
  - 选中文字, `d` 删除 或者 `c` 改变
- `u` 撤销, `<C-r>` 重做
- `y` 复制 （ “yank”） 
- `p` 粘贴
- `~` 改变字符的大小写



### 计数

可以用一个计数来结合“名词”和“动词”，这会执行指定操作若干次。

- `3w` 向前移动三个词
- `5j` 向下移动5行
- `7dw` 删除7个词



### 修饰语

可以用修饰语改变“名词”的意义。修饰语有 `i`，表示“内部”或者“在内”，和 `a`， 表示“周围”。

- `ci(` 改变当前括号内的内容
- `ci[` 改变当前方括号内的内容
- `da'` 删除一个单引号字符串， 包括周围的单引号





## 数据整理

### 简介

将某种格式存储的数据转换成另外一种格式

使用管道运算符也是数据整理

日志处理，每行的内容大概为

```
Jan 17 03:13:00 thesquareplanet.com sshd[2631]: Disconnected from invalid user wdl 46.97.239.16 port 55920 [preauth]
```

将一个远程服务器上的文件传递给本机的 `grep` 程序，看哪些用户曾经尝试过登录我们的服务器：

```
ssh myserver 'journalctl | grep sshd | grep "Disconnected from"' | less
```

（可以把数据来源`ssh myserver 'journalctl`换为`cat /var/log/messages`）

`less` ：创建一个文件分页器

`sed` 是一个基于文本编辑器`ed`构建的”流编辑器” 

替换命令`s`：`s/REGEX/SUBSTITUTION/`, 其中 `REGEX` 部分是我们需要使用的正则表达式，而 `SUBSTITUTION` 是用于替换匹配结果的文本

```
ssh myserver journalctl
 | grep sshd
 | grep "Disconnected from"
 | sed 's/.*Disconnected from //'
```



### 正则表达式

- `.` 除换行符之外的”任意单个字符”
- `*` 匹配前面字符零次或多次
- `+` 匹配前面字符一次或多次
- `[abc]` 匹配 `a`, `b` 和 `c` 中的任意一个
- `(RX1|RX2)` 任何能够匹配`RX1` 或 `RX2`的结果
- `^` 行首
- `$` 行尾

`sed` 的正则表达式有些时候需要在这些模式前添加`\`才能使其具有特殊含义。或者也可以添加`-E`选项来支持这些匹配

`*` 和 `+` 在默认情况下是贪婪模式，增加一个`?` 后缀使其变成非贪婪模式

```
| sed -E 's/.*Disconnected from (invalid |authenticating )?user (.*) [^ ]+ port [0-9]+( \[preauth\])?$/\2/'
```

`[^ ]+` 会匹配任意非空且不包含空格的序列

希望能够将用户名保留下，可以使用“捕获组”来完成。被圆括号内的正则表达式匹配到的文本，都会被存入一系列以编号区分的捕获组中，捕获组的内容可以在替换字符串时使用，例如`\1`、 `\2`、`\3`



### sed数据整理

```
ssh myserver journalctl
 | grep sshd
 | grep "Disconnected from"
 | sed -E 's/.*Disconnected from (invalid |authenticating )?user (.*) [^ ]+ port [0-9]+( \[preauth\])?$/\2/'
 | sort | uniq -c
 | sort -nk1,1 | tail -n10
```

`sort` 会对其输入数据进行排序。

`uniq -c` 会把连续出现的行折叠为一行并使用出现次数作为前缀

`sort -n` 会按照数字顺序对输入进行排序

`-k1,1` 则表示“仅基于以空格分割的第一列进行排序”

`n` 部分表示“仅排序到第n个部分”，默认情况是到行尾

可以使用 `head` 来代替`tail`。或者使用`sort -r`来进行倒序排序



### awk

`awk` 其实是一种编程语言

```
 | sort | uniq -c
 | sort -nk1,1 | tail -n10
 | awk '{print $2}' | paste -sd,
```

只想获取用户名，而且不要一行一个地显示

 `paste`命令来合并行(`-s`)，并指定一个分隔符进行分割 (`-d`)

`$0` 表示整行的内容，`$1` 到 `$n` 为一行中的 n 个区域

区域的分割基于 `awk` 的域分隔符（默认是空格，可以通过`-F`来修改）

```
 | awk '$1 == 1 && $2 ~ /^c[^ ]*e$/ { print $2 }' | wc -l
```

统计所有以`c` 开头，以 `e` 结尾，并且仅尝试过一次登录的用户

 `wc -l` 统计输出结果的行数，`-c`bytes数，`-w`字数

既然 `awk` 是一种编程语言，那么则可以这样：

```
BEGIN { rows = 0 }
$1 == 1 && $2 ~ /^c[^ ]*e$/ { rows += $1 }
END { print rows }
```



### 分析数据

将每行的数据加起来：

```
 | paste -sd+ | bc -l
```

`bc`即伯克利计算器

如果已经安装了R语言：

```
| R --slave -e 'x <- scan(file="stdin", quiet=TRUE); summary(x)'
```

`summary` 可以打印某个向量的统计结果。我们将输入的一系列数据存放在一个向量后，利用R语言就可以得到我们想要的统计数据

绘制一些简单的图表：

```
| sort | uniq -c
 | sort -nk1,1 | tail -n10
 | gnuplot -p -e 'set boxwidth 0.5; plot "-" using 1:xtic(2) with boxes'
```

目前为止的讨论都是基于文本数据，但对于二进制文件其实同样有用。例如可以用 ffmpeg 从相机中捕获一张图片，将其转换成灰度图后通过SSH将压缩后的文件发送到远端服务器，并在那里解压、存档并显示

```
ffmpeg -loglevel panic -i /dev/video0 -frames 1 -f image2 -
 | convert - -colorspace gray -
 | gzip
 | ssh mymachine 'gzip -d | tee copy.jpg | env DISPLAY=:0 feh -'
```





## 命令行环境

### 任务控制

大多数情况下，我们可以使用 `Ctrl-C` 来停止命令的执行

shell 会使用 UNIX 提供的信号机制执行进程间通信。当输入 `Ctrl-C` 时，shell 会发送一个`SIGINT` 信号到进程。

`Ctrl-\`可以发送`SIGQUIT` 信号

`SIGTERM` 则是一个更加通用的也更加优雅的退出信号。为了发出这个信号需要使用 `kill` 命令, 语法是： `kill -TERM <PID>`

```
kill -STOP %1
```

`SIGSTOP` 会让进程暂停。在终端中，键入 `Ctrl-Z` 会让 shell 发送 `SIGTSTP` 信号，`SIGTSTP`是 Terminal Stop 的缩写（即`terminal`版本的`SIGSTOP`）

 `fg`或 `bg`命令恢复暂停的工作。它们分别表示在前台继续或在后台继续

`jobs`命令会列出当前终端会话中尚未完成的全部任务。可以使用 pid 引用这些任务（也可以用 `pgrep`找出 pid）

命令中的 `&` 后缀可以让命令在直接在后台运行，这使得可以直接在 shell 中继续做其他操作，不过它此时还是会使用 shell 的标准输出

让已经在运行的进程转到后台运行，可以键入`Ctrl-Z` ，然后紧接着再输入`bg`

后台的进程仍然是终端进程的子进程，一旦关闭终端，会发送另外一个信号`SIGHUP`，这些后台的进程也会终止。为了防止这种情况发生，可以使用 `nohup`(一个用来忽略 `SIGHUP` 的封装) 来运行程序

```
nohup sleep 2000 &
```

`SIGKILL` 是一个特殊的信号，它不能被进程捕获并且它会马上结束该进程

使用`man signal`获取更多信息



### 终端多路复用

同时执行多个任务，例如同时运行编辑器，并在终端的另外一侧执行程序

`tmux` 终端多路复用器

快捷键都是类似 `<C-b> x` 这样的组合，即需要先按下`Ctrl+b`，松开后再按下 `x`

`tmux` 中对象的继承结构如下：

- 会话：每个会话都是一个独立的工作区，其中包含一个或多个窗口
  - `tmux` 开始一个新的会话
  - `tmux new -s NAME` 以指定名称开始一个新的会话
  - `tmux ls` 列出当前所有会话
  - 在 `tmux` 中输入 `<C-b> d` ，将当前会话分离
  - `tmux a` 重新连接最后一个会话。您也可以通过 `-t` 来指定具体的会话
- 窗口：相当于编辑器或是浏览器中的标签页，从视觉上将一个会话分割为多个部分
  - `<C-b> c` 创建一个新的窗口，使用 `<C-d>`关闭
  - `<C-b> N` 跳转到第 *N* 个窗口，注意每个窗口都是有编号的
  - `<C-b> p` 切换到前一个窗口
  - `<C-b> n` 切换到下一个窗口
  - `<C-b> ,` 重命名当前窗口
  - `<C-b> w` 列出当前所有窗口
- 面板：像 vim 中的分屏一样，面板使我们可以在一个屏幕里显示多个 shell
  - `<C-b> "` 水平分割
  - `<C-b> %` 垂直分割
  - `<C-b> <方向>` 切换到指定方向的面板，<方向> 指的是键盘上的方向键
  - `<C-b> z` 切换当前面板的缩放
  - `<C-b> [` 开始往回卷动屏幕。您可以按下空格键来开始选择，回车键复制选中的部分
  - `<C-b> <空格>` 在不同的面板排布间切换



### 别名

别名相当于一个长命令的缩写，shell 会自动将其替换成原本的命令

```
alias ll="ls -lh"
alias sl=ls
alias mv="mv -i"
alias ll  # 获取别名的定义
```

 shell 并不会保存别名。为了让别名持续生效，需要将配置放进 shell 的启动文件里，像是`.bashrc` 或 `.zshrc`



### 配置文件（Dotfiles）

很多程序的配置都是通过纯文本格式的被称作点文件的配置文件来完成的（之所以称为点文件，是因为它们的文件名以 `.` 开头，例如 `~/.vimrc`。也正因为此，它们默认是隐藏文件，`ls`并不会显示它们）

- `bash` - `~/.bashrc`, `~/.bash_profile`
- `git` - `~/.gitconfig`
- `vim` - `~/.vimrc` 和 `~/.vim` 目录
- `ssh` - `~/.ssh/config`
- `tmux` - `~/.tmux.conf`

很多程序都要求在 shell 的配置文件中包含一行类似 `export PATH="$PATH:/path/to/program/bin"` 的命令，这样才能确保这些程序能够被 shell 找到

如何管理这些配置文件：它们应该在它们的文件夹下，并使用git进行管理，然后通过脚本将其 **符号链接** 到需要的地方



### 远端设备

可以使用 `ssh` 连接到其他服务器：

```
ssh foo@bar.mit.edu
```

这里尝试以用户名 `foo` 登录服务器 `bar.mit.edu`

服务器可以通过 URL 指定（例如`bar.mit.edu`），也可以使用 IP 指定（例如`foobar@192.168.1.42`）

可以直接远程执行命令。 `ssh foobar@server ls` 可以直接在用foobar的命令下执行 `ls` 命令

`ssh foobar@server ls | grep PATTERN` 会在本地查询远端 `ls` 的输出而 `ls | ssh foobar@server grep PATTERN` 会在远端对本地 `ls` 输出的结果进行查询



基于密钥的验证机制使用了密码学中的公钥，只需要向服务器证明客户端持有对应的私钥，而不需要公开其私钥，这样就可以避免每次登录都输入密码的麻烦



使用 ssh 复制文件有很多方法：

- `ssh+tee`, 最简单的方法是执行 `ssh` 命令，然后通过这样的方法利用标准输入实现 `cat localfile | ssh remote_server tee serverfile`。`tee` 命令会将标准输出写入到一个文件；
- `scp` ：当需要拷贝大量的文件或目录时，使用`scp` 命令则更加方便，因为它可以方便的遍历相关路径。语法如下：`scp path/to/local_file remote_host:path/to/remote_file`
- `rsync` 对 `scp` 进行了改进，它可以检测本地和端的文件以防止重复拷贝。它还可以提供一些诸如符号连接、权限管理等精心打磨的功能。甚至还可以基于 `--partial`标记实现断点续传 ，语法和`scp`类似
- 

本地端口转发，即远端设备上的服务监听一个端口，而在本地设备上的一个端口建立连接并转发到远程端口上

例如，在远端服务器上运行 Jupyter notebook 并监听 `8888` 端口。 然后，建立从本地端口 `9999` 的转发，使用 `ssh -L 9999:localhost:8888 foobar@remote_server` 。这样只需要访问本地的 `localhost:9999` 即可



SSH 配置，为它们创建一个别名

```
alias my_server="ssh -i ~/.id_ed25519 --port 2222 -L 9999:localhost:8888 foobar@remote_server
```

更好的方法是使用 `~/.ssh/config`

```
Host vm
    User foobar
    HostName 172.16.174.141
    Port 2222
    IdentityFile ~/.ssh/id_ed25519
    LocalForward 9999 localhost:8888

# 在配置文件中也可以使用通配符
Host *.mit.edu
    User foobaz
```

`~/.ssh/config` 文件也可以被当作配置文件，而且一般情况下也是可以被导入其他配置文件的。如果将其公开到互联网上，可能造成信息泄露

服务器侧的配置通常放在 `/etc/ssh/sshd_config`，可以在这里配置免密认证、修改 ssh 端口、开启 X11 转发等等





## 版本控制(Git）

### 简介

版本控制系统 (VCSs) 是一类用于追踪源代码（或其他文件、文件夹）改动的工具。VCS通过一系列的快照将某个文件夹及其内容保存了起来，每个快照都包含了文件或文件夹的完整状态。同时还维护了快照创建者的信息以及每个快照的相关信息等等。



### Git 的数据模型

Git 将顶级目录中的文件和文件夹作为集合，并通过一系列快照来管理其历史记录。在Git的术语里，文件被称作Blob对象（数据对象），也就是一组数据。目录则被称之为“树”，它将名字与 Blob 对象或树对象进行映射（使得目录中可以包含其他目录）。快照则是被追踪的最顶层的树。例如，一个树看起来可能是这样的：

```
<root> (tree)
|
+- foo (tree)
|  |
|  + bar.txt (blob, contents = "hello world")
|
+- baz.txt (blob, contents = "git is wonderful")
```

在 Git 中，历史记录是一个由快照组成的有向无环图，快照被称为“提交”（commit）。通过可视化的方式来表示这些历史提交记录时，看起来是这样的：

```
o <-- o <-- o <-- o <---- o
            ^            /
             \          v
              --- o <-- o
```

Git 中的提交是不可改变的。但这并不代表错误不能被修改，只不过这种“修改”实际上是创建了一个全新的提交记录。而引用则被更新为指向这些新的提交。

以伪代码的形式来学习 Git 的数据模型，可能更加清晰：

```
// 文件就是一组数据
type blob = array<byte>

// 一个包含文件和目录的目录
type tree = map<string, tree | blob>

// 每个提交都包含一个父辈，元数据和顶层树
type commit = struct {
    parent: array<commit>
    author: string
    message: string
    snapshot: tree
}
```

这是一种简洁的历史模型。



Git 中的对象可以是 blob、树或提交：

```
type object = blob | tree | commit
```

Git 在储存数据时，所有的对象都会基于它们的 SHA-1 哈希进行寻址。

Blobs、树和提交都一样，它们都是对象。当它们引用其他对象时，它们并没有真正的在硬盘上保存这些对象，而是仅仅保存了它们的哈希值作为引用。

例如，上面例子中的树（可以通过 `git cat-file -p 698281bc680d1995c5f4caaf3359721a5a58d48d` 来进行可视化），看上去是这样的：

```
100644 blob 4448adbf7ecd394f42ae135bbeed9676e894af85    baz.txt
040000 tree c68d233a33c5c06e0340e4c224f0afca87c8ce87    foo
```

Git给这些哈希值赋予人类可读的名字，也就是引用（references）。引用是指向提交的指针。与对象不同的是，它是可变的（引用可以被更新，指向新的提交）。例如，`master` 引用通常会指向主分支的最新一次提交。

在 Git 中，我们当前的位置有一个特殊的索引，它就是 “HEAD”。



最后，我们可以粗略地给出 Git 仓库的定义了：对象和引用。

在硬盘上，Git 仅存储对象和引用：因为其数据模型仅包含这些东西。所有的 `git` 命令都对应着对提交树的操作，例如增加对象，增加或删除引用。



Git 使用一种叫做 “暂存区（staging area）”的机制，它允许指定下次快照中要包括那些改动。



### Git 的命令行接口

强烈推荐阅读 [Pro Git 中文版](https://git-scm.com/book/zh/v2)。

#### 基础

- `git help <command>`: 获取 git 命令的帮助信息
- `git init`: 创建一个新的 git 仓库，其数据会存放在一个名为 `.git` 的目录下
- `git status`: 显示当前的仓库状态
- `git add <filename>`: 添加文件到暂存区
- `git commit`: 创建一个新的提交
- `git log`: 显示历史日志
- `git log --all --graph --decorate`: 可视化历史记录（有向无环图）
- `git diff <filename>`: 显示与暂存区文件的差异
- `git diff <revision> <filename>`: 显示某个文件两个版本之间的差异
- `git checkout <revision>`: 更新 HEAD 和目前的分支

#### 分支和合并

- `git branch`: 显示分支
- `git branch <name>`: 创建分支
- `git checkout -b <name>`: 创建分支并切换到该分支
  - 相当于 `git branch <name>; git checkout <name>`
- `git merge <revision>`: 合并到当前分支
- `git mergetool`: 使用工具来处理合并冲突
- `git rebase`: 将一系列补丁变基（rebase）为新的基线

#### 远端操作

- `git remote`: 列出远端
- `git remote add <name> <url>`: 添加一个远端
- `git push <remote> <local branch>:<remote branch>`: 将对象传送至远端并更新远端引用
- `git branch --set-upstream-to=<remote>/<remote branch>`: 创建本地和远端分支的关联关系
- `git fetch`: 从远端获取对象/索引
- `git pull`: 相当于 `git fetch; git merge`
- `git clone`: 从远端下载仓库

#### 撤销

- `git commit --amend`: 编辑提交的内容或信息

- `git reset HEAD <file>`: 恢复暂存的文件

- `git checkout -- <file>`: 丢弃修改

- `git restore`: git2.32版本后取代git reset 进行许多撤销操作

  

### Git 高级操作

- `git config`: Git 是一个高度可定制的工具

- `git clone --depth=1`: 浅克隆（shallow clone），不包括完整的版本历史信息

- `git add -p`: 交互式暂存

- `git rebase -i`: 交互式变基

- `git blame`: 查看最后修改某行的人

- `git stash`: 暂时移除工作目录下的修改内容

- `git bisect`: 通过二分查找搜索历史记录

- `.gitignore`: 指定故意不追踪的文件

  



## 调试及性能分析

### 打印调试法与日志

执行 `echo -e "\e[38;2;255;0;0mThis is red\e[0m"` 会打印红色的字符串：`This is red`

程序的日志通常存放在 `/var/log`

系统开始使用 **system log**，所有的日志都会保存在这里。`systemd` 会将日志以某种特殊格式存放于`/var/log/journal`，您可以使用 `journalctl`命令显示这些消息，像 `lnav`这样的工具，它为日志文件提供了更好的展现和浏览方式

可以使用`dmesg` 命令来读取内核的日志

将日志加入到系统日志中

```
logger "Hello Logs"
journalctl --since "1m ago" | grep Hello
```



### 调试器

一种可以允许我们和正在执行的程序进行交互的程序

Python 的调试器是ipdb

- **l**(ist) - 显示当前行附近的11行或继续执行之前的显示；
- **s**(tep) - 执行当前行，并在第一个可能的地方停止；
- **n**(ext) - 继续执行直到当前函数的下一条语句或者 return 语句；
- **b**(reak) - 设置断点（基于传入的参数）；
- **p**(rint) - 在当前上下文对表达式求值并打印结果。还有一个命令是**pp** ，它使用 `pprint`打印；
- **r**(eturn) - 继续执行直到当前函数返回；
- **q**(uit) - 退出调试器。

对于更底层的编程语言：gdb

当程序需要执行一些只有操作系统内核才能完成的操作时，它需要使用 系统调用。有一些命令可以追踪程序执行的系统调用，如`strace`

```
sudo strace -e lstat ls -l > /dev/null
```



### 静态分析

有些问题是不需要执行代码就能发现的，可以用静态分析工具`pyflakes`或`mypy`

之前介绍的 `shellcheck`是类似的工具，但它是应用于 shell 脚本的

大多数的编辑器和 IDE 都支持在编辑界面显示这些工具的分析结果、高亮有警告和错误的位置。 这个过程通常称为 `code linting`

在 vim 中，有 `ale` 或 `syntastic`可以做同样的事情。 在 Python 中， `pylint` 和 `pep8`是两种用于进行风格检查的工具



### 性能分析工具

通常指的是 CPU 性能分析工具。 CPU 性能分析工具有两种： 追踪分析器（*tracing*）及采样分析器（*sampling*）

追踪分析器 会记录程序的每一次函数调用，而采样分析器则只会周期性的监测（通常为每毫秒）程序并记录程序堆栈

使用 `cProfile` 模块来分析每次函数调用所消耗的时间

```
python -m cProfile -s tottime grep.py 1000 '^(import|\s*def)[^,]*$' *.py
```

更加符合直觉的显示分析信息的方式是包括每行代码的执行时间，这也是行分析器的工作，基于行来显示时间

```
kernprof -l -v a.py
```

应对内存类的bug ：`memory-profiler`（和 `line-profiler` 类似）

```
python -m memory_profiler example.py
```



在我们使用`strace`调试代码的时候，可能会希望忽略一些特殊的代码并希望在分析时将其当作黑盒处理。`perf`] 命令将 CPU 的区别进行了抽象，它不会报告时间和内存的消耗，而是报告与程序相关的系统事件

- `perf list` - 列出可以被 pref 追踪的事件；
- `perf stat COMMAND ARG1 ARG2` - 收集与某个进程或指令相关的事件；
- `perf record COMMAND ARG1 ARG2` - 记录命令执行的采样信息并将统计数据储存在`perf.data`中；
- `perf report` - 格式化并打印 `perf.data` 中的数据。



可视化方面，对于采样分析器来说，常见的显示 CPU 分析数据的形式是火焰图，火焰图会在 Y 轴显示函数调用关系，并在 X 轴显示其耗时的比例

调用图和控制流图可以显示子程序之间的关系，可以使用 `pycallgraph` 来生成



类似 `hyperfine`这样的命令行可以快速进行基准测试

例如比较`fd` 和 `find`

```
hyperfine --warmup 3 'fd -e jpg' 'find . -iname "*.jpg"'
```



### 资源监控

有时候，分析程序性能的第一步是搞清楚它所消耗的资源。程序变慢通常是因为它所需要的资源不够了

- **通用监控** - 最流行的工具是`htop`
- **I/O 操作** - `iotop`
- **磁盘使用** - `df`
- **内存使用** - `free`
- **打开文件** - `lsof`
- **网络连接和配置** - `ss`
- **网络使用** - `nethogs`和 `iftop`
