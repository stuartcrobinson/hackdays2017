<template>
  <div class="container0">
    <div>
      Counter: {{ $store.state.count }} times, count is {{ evenOrOdd }}.
      <br>
      Answer: {{ $store.state.isyes }}
      <br>
      gifurl: {{ $store.state.gifurl }}
      <br>
      <img :src="$store.state.gifurl"/>

    </div>
    <div class="container">
      <div class="row text-center">
        Clicked: {{ $store.state.count }} times, count is {{ evenOrOdd }}
        <button class="btn btn-success" @click="increment">+</button>
        <button class="btn btn-danger" @click="decrement">-</button>
        <button class="btn" @click="incrementIfOdd">Increment if odd</button>
        <button class="btn" @click="incrementAsync">Increment async</button>
        <button class="btn" @click="yesnoaction">yesnoaction</button>
        <br>
        <search-field-container/>
        is loading? {{ $store.state.isLoadingProductsSearch }}
        <br>
        length? {{ $store.state.queriedProducts.length }}

        <card-list :products="$store.state.queriedProducts"/>

      </div>
    </div>
  </div>
</template>

<!--<script>-->
<!--import { mapGetters, mapActions } from 'vuex'-->

<!--export default {-->
<!--name: 'hello',-->
<!--computed: mapGetters([-->
<!--'evenOrOdd'-->
<!--]),-->
<!--methods: mapActions([-->
<!--'increment',-->
<!--'decrement',-->
<!--'incrementIfOdd',-->
<!--'incrementAsync'-->
<!--])-->
<!--}-->
<!--</script>-->

<script>
  import { mapGetters, mapActions } from 'vuex'
  import Example1 from './Example1.vue'
  import Card from './Card.vue'
  import CardList from './CardList.vue'
  import SearchFieldContainer from './SearchFieldContainer.vue'
  import ItemTemplate from './ItemTemplate.vue'

  // https://github.com/vuejs/vuex/issues/367
  // this.$store.dispatch('updatecustomer', this.custumer)

  export default {
    name: 'hello',
    components: {
      Example1, Card, SearchFieldContainer, CardList, ItemTemplate
    },
    computed: mapGetters([
      'evenOrOdd'
    ]),
    methods: {
      ...mapActions([
        'increment',
        'decrement',
        'incrementIfOdd',
        'incrementAsync',
        'yesnoaction',
        'loadproducts'
      ])
    },
    data () {
      return {
        msg: 'Welcome to Your Vue.js App',
        age: 100,
        item: 'firstitem?',
        items: [],
        template: ItemTemplate
      }
    }
  }
</script>

