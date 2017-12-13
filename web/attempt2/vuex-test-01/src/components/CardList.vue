<template>
  <div class="flex-container">
    <br>
    <div class="flex-item" :style="isLoading ? '': 'display: none;'">
      <card imgUrl="https://i.allthepics.net/2017/08/06/fidgit-spinner-dragonb1f6d.gif"
            productTitle="Loading..."
            productId="...">
      </card>
    </div>
    <div class="flex-item" v-for="product in products">
      <card
        :imgUrl="product.productImageUrl"
        :productTitle="product.productTitle"
        productDescription='no description'
        :productId="product.productId"
        :theclick="clickToRemove ? removeProduct : browse"
        :hoverOverlay="clickToRemove ? '❌' : ''"
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
    props: ['products', 'isLoading', 'clickToRemove'],
    methods: {
      ...mapActions([
        'browse',
        'removeProduct'
      ])
    }
  }
</script>

<style>
  .flex-container {
    /*width: 100%;*/
    display: flex;
    /*flex-flow: row wrap;*/
    justify-content: left;
    overflow-x: scroll;
    overflow-y: hidden;
    padding-bottom: 10px;
  }

  .flex-item {
    padding-left: 5px;
    padding-right: 5px;
  }
</style>

<!--methods: {-->
<!--myclick (productId) {-->
<!--this.$store.dispatch('browse', productId)-->
<!--}-->
<!--},-->
