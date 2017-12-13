<template>

  <div class="container" :style="productTitle == undefined? 'display: none;' : ''" @click="theclick(stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object)">
    <img height="140" :src="imgUrl"/>
    <div class="centered" id="redx"><p>{{ hoverOverlay }}</p></div>

    <br>
    <strong>{{ productTitle }}</strong>
    <!--<br>-->
    <!--{{ productId }}-->

  </div>

</template>

<script>
  export default {
    props: ['hoverOverlay', 'imgUrl', 'isHidden', 'productTitle', 'productDescription', 'productId', 'theclick', 'stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object']
  }
</script>

<style>
  /* Container holding the image and the text */
  .container {
    position: relative;
    text-align: center;
    cursor: pointer;
    /*height:200px;*/
    min-width:150px;
    max-height:180px;
    /*max-height:200px;*/
  }

  /* Centered text */
  .centered {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 100px;
    cursor: pointer
  }

  #redx {
    display: none;
  }

  .container:hover #redx {
    display: block;
  }

</style>

<!--style="cursor:pointer"-->
