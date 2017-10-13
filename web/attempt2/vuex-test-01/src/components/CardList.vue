<template>
  <div class="flex-container">
    <br>
    <div class="flex-item" v-for="item in products">
      <card :imgUrl="item._source.imageUrl"
            :productTitle="item._source.title"
            :productDescription="item._source.description"
            :productId="item._source.productId"></card>
    </div>
  </div>
</template>

<script>
  import Card from './Card.vue'

  export default {
    components: {Card},
    props: ['products']
  }
</script>

<style>
  .flex-container {
    display: flex;
    flex-flow: row wrap;
    justify-content: space-around;
  }
</style>
