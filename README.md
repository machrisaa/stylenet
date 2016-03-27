# Stylenet

Neural Network with Style Synthesis based on 2 major methods [Gram matrix](http://arxiv.org/abs/1508.06576) and [Markov Random Fields](http://arxiv.org/abs/1601.04589). By given a content and style image, the style and pattern can be transfered to paint the content.

<table>
  <tr>
    <td><img src="https://github.com/machrisaa/stylenet/blob/master/images/cat-water-colour.jpg"/></td>
  </tr>
  <tr>
    <td>Content</td>
  </tr>
</table>

##Requirement
[Tensorflow](https://www.tensorflow.org/versions/r0.7/get_started/index.html)
[Tensorflow-VGG](https://github.com/machrisaa/tensorflow-vgg)

##Basic Usage
```
stylenet_patch.render_gen( <content image path> , <style image path>, height=<output height>)
```
