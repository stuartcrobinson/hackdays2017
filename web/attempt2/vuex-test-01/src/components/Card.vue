<template>
  <div>
    <div class="container" :style="productTitle == undefined? 'display: none;' : ''" @click="theclick(stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object)">
      <img height="140" :src="imgUrl" :alt="productId"/>
      <div class="centered" id="redx"><p>{{ hoverOverlay }}</p></div>

      <br>
      <strong>{{ productTitle }}</strong>
    </div>

    <!--<div class="tooltip">Hover over me-->
      <!--<span class="tooltiptext">Tooltip text</span>-->
    <!--</div>-->
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
    min-width: 150px;
    max-height: 180px;
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

  .tooltip {
    position: relative;
    display: inline-block;
    border-bottom: 1px dotted black;
  }

  .tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -60px;
    opacity: 0;
  }

  .tooltip .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #555 transparent transparent transparent;
  }

  .tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
  }

</style>

<!--style="cursor:pointer"-->
