<template>
  <div class="flex-container">
    <br>
    <div class="flex-item" v-for="product in products">
      <card
        :imgUrl="product.productImageUrl"
        :productTitle="product.productTitle"
        productDescription='no description'
        :productId="product.productId"
        :theclick="browse"
        :stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object="product">
      </card>
    </div>
  </div>
</template>

<script>
  import Card from './Card.vue'
  import { mapActions } from 'vuex'

  export default {
    components: {Card},
    props: ['products'],
    methods: {
      ...mapActions([
        'browse'
      ])
    }
  }
</script>

<style>
  .flex-container {
    display: flex;
    flex-flow: row wrap;
    justify-content: space-around;
  }
  .flex-item {
    padding-left:5px;
    padding-right:5px;
  }
</style>

<!--methods: {-->
<!--myclick (productId) {-->
<!--this.$store.dispatch('browse', productId)-->
<!--}-->
<!--},-->
