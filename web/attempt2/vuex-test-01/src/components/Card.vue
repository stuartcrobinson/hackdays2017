<template>
  <div id="example1">
    productId: {{ productId }}
    <br>
    productTitle: {{ productTitle }}
    <br>
    productDescription: {{ productDescription }}
    <!--<br>-->
    <!--imgurl: {{ imgUrl }}-->
    <br>
    <img :src="imgUrl"/>
  </div>
</template>

<script>
  export default {
    props: ['imgUrl', 'productTitle', 'productDescription', 'productId']
  }
</script>
