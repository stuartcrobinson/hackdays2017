<template>
  <div>
    <strong>{{ productTitle }}</strong>
    <br>
    <img :src="imgUrl"/>
  </div>
</template>

<script>
  export default {
    props: ['imgUrl', 'productTitle', 'productDescription', 'productId']
  }
</script>
