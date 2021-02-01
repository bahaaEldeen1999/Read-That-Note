# :musical_score: Read That Note

<img src="./imgs/img.png"><br>
<div align="center">
  
  [![GitHub contributors](https://img.shields.io/github/contributors/bahaaEldeen1999/Read-That-Note)](https://github.com/bahaaEldeen1999/Read-That-Note/contributors)
  [![GitHub issues](https://img.shields.io/github/issues/bahaaEldeen1999/Read-That-Note)](https://github.com/bahaaEldeen1999/Read-That-Note/issues)
  [![GitHub forks](https://img.shields.io/github/forks/bahaaEldeen1999/Read-That-Note)](https://github.com/bahaaEldeen1999/Read-That-Note/network)
  [![GitHub stars](https://img.shields.io/github/stars/bahaaEldeen1999/Read-That-Note)](https://github.com/bahaaEldeen1999/Read-That-Note/stargazers)
  [![GitHub license](https://img.shields.io/github/license/bahaaEldeen1999/Read-That-Note)](https://github.com/bahaaEldeen1999/Read-That-Note/blob/master/LICENSE)
  <img src="https://img.shields.io/github/languages/count/bahaaEldeen1999/Read-That-Note" />
  <img src="https://img.shields.io/github/languages/top/bahaaEldeen1999/Read-That-Note" />
  <img src="https://img.shields.io/github/languages/code-size/bahaaEldeen1999/Read-That-Note" />
  <img src="https://img.shields.io/github/issues-pr-raw/bahaaEldeen1999/Read-That-Note" />

</div>
### Read That Note is an optical musical recognition system, where the input is a musical sheet score, and the output is a text representing each note and symbol

## methodology

<ol>
<li>Deskewing the Image</li>
<li>Convert Image To Binary</li>
<li>Remove Staff Lines</li>
<li>Remove Musical Notes</li>
<li>Extract Each 5 Staves Together</li>
<li>Musical Note Segmentation</li>
<li>Remove Unwanted labels</li>
<li>Classify Musical Note</li>
<li>Convert Line Array To Usable Data</li>
<li>Output To File</li>
</ol>

### for more details on the used algorithm please follow this link: <a href="https://github.com/bahaaEldeen1999/Read-That-Note/tree/main/methodology">Methodology</a>

## Experiment

### input

<img src="./imgs/0.png">

### staff line removel

<img src="./imgs/1.png">

### staff lines without musical notes

<img src="./imgs/2.png">

### sine segmentation

<img src="./imgs/3.png">

<img src="./imgs/4.png">

<img src="./imgs/5.png">

### character segmentation

<div style="text-align:center">
<img src="./imgs/6.png" style="margin:20px">

<img src="./imgs/7.png" style="margin:20px">
<img src="./imgs/8.png" style="margin:20px">
<img src="./imgs/9.png" style="margin:20px">
<img src="./imgs/10.png " style="margin:20px"><br>
<img src="./imgs/11.png" style="margin:20px">
<img src="./imgs/12.png" style="margin:20px">
<img src="./imgs/13.png" style="margin:20px">
<img src="./imgs/14.png" style="margin:20px">
<img src="./imgs/15.png" style="margin:20px"><br>
<img src="./imgs/16.png" style="margin:20px">
<img src="./imgs/17.png" style="margin:20px">
<img src="./imgs/18.png" style="margin:20px">
<img src="./imgs/19.png" style="margin:20px">
<img src="./imgs/20.png" style="margin:20px">
</div>

### output

<h3>
{<br>
[ \meter<"4/4"> d1/4 e1/32 e1/8 e1/32 e1/32{e1/4,g1/4}  e1/4 e1/16 c1/16 g1/32 c1/4 e1/32],<br>
[ \meter<"4/4">{b1/4,e1/4,g1/4}  a1/8 d1/8 c1/16 g1/16 d1/16 e1/16 c2/16 g2/16 d2/16 e2/16{b1/4,f1/4,g1/4}  c1/4 a1/4 a1/8 a1/32],<br>
[ \meter<"4/4"> e1/16 e1/16 e1/16 e1/16 e1/4 e#1/4 g1/4 g&&1/4 g1/4 e#2/4]<br>
}
</h3>
