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

        <p>{{ namee }} is {{ age }} years old.</p>
        <br>
        <search-field/>
        is loading? {{ $store.state.isLoadingProductsSearch }}
        <!--<br>-->
        <!--products? {{ $store.state.queriedProducts }}-->
        <br>
        length? {{ $store.state.queriedProducts.length }}

        <card-list :products="$store.state.queriedProducts"/>

        <ol id="example-1">
          <li v-for="item in $store.state.queriedProducts">

            <!--<card :productTitle="item._source.title"></card>-->

            <!--<br>-->
            <!--{{item}}-->
            <!--<br>-->
            <card :imgUrl="item._source.imageUrl"
                  :productTitle="item._source.title"
                  :productDescription="item._source.description"
                  :productId="item._source.productId"></card>
            <!--{{ item }}-->
          </li>
        </ol>
      </div>
    </div>
    <example1 nickname="bob">hi</example1>
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
  import SearchField from './SearchField.vue'
  import ItemTemplate from './ItemTemplate.vue'

  // https://github.com/vuejs/vuex/issues/367
  // this.$store.dispatch('updatecustomer', this.custumer)

  export default {
    name: 'hello',
    components: {
      Example1, Card, SearchField, CardList, ItemTemplate
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
        namee: 'Ashley',
        age: 100,
        item: 'firstitem?',
        items: [],
        template: ItemTemplate
      }
    }
  }
</script>

