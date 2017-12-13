<template>
  <div class="container0">
    <div id="example crap" style="display: none">
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
        <br class="row text-center">
        Clicked: {{ $store.state.count }} times, count is {{ evenOrOdd }}
        <button class="btn btn-success" @click="increment">+</button>
        <button class="btn btn-danger" @click="decrement">-</button>
        <button class="btn" @click="incrementIfOdd">Increment if odd</button>
        <button class="btn" @click="incrementAsync">Increment async</button>
        <button class="btn" @click="yesnoaction">yesnoaction</button>
        <button class="btn" @click="debouncedMethod">debouncedMethod</button>
        <button class="btn" @click="throttledMethod">throttledMethod</button>
        <button class="btn" @click="otherMethod">otherMethod</button>
        <br>
      </div>
    </div>

    <search-field-container/>

    <br>

    <table>
      <tr>
        <th><span style="font-size:40px">Current Product</span></th>
        <th><span style="font-size:40px">Browse History</span></th>
      </tr>
      <tr>
        <td style="border:4px solid blue; padding:10px">
          <!--<strong>current product</strong>-->
          <card :imgUrl="$store.state.currentProduct.productImageUrl"
                :productTitle="$store.state.currentProduct.productTitle"
                productDescription='no description'
                :productId="$store.state.currentProduct.productId"
                :theclick="browse"
                :stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object="$store.state.currentProduct">
          </card>
        </td>
        <td style="border:1px solid black; padding:10px">
          <!--<strong>browse history</strong><br/>-->
          <card-list :products="$store.state.browsedProducts"/>
        </td>
      </tr>
    </table>

    <hr>

    <strong><span style="font-size:40px">Search Results</span></strong>
    is loading? {{ $store.state.isLoadingProductsSearch }}  -- num results: {{ $store.state.queriedProducts.length }}
    <card-list :products="$store.state.queriedProducts"/>

    <hr>
    <strong><span style="font-size:40px">Bronto Recs</span></strong>
    is loading? {{ $store.state.isLoadingBrontoRecommendedProducts }}
    <br/>
    <card-list :products="$store.state.brontoRecommendedProducts"/>

    <hr>
    <strong><span style="font-size:40px">Neural Network Recs</span></strong>
    is loading? {{ $store.state.isLoadingStuartRecommendedProducts }}
    <br/>
    <card-list :products="$store.state.stuartRecommendedProducts"/>

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
  import _ from 'lodash'

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
        'loadproducts',
        'browse'
      ]),
      dostuff () {
        console.log('fired')
      },
      throttledMethod: _.throttle(function () {
        console.log('I get fired every two seconds!')
      }, 2000),
      debouncedMethod: _.debounce(function () {
        console.log('debouncedMethod')
      }, 2000),
      otherMethod () {
        this.debouncedMethod()
      }
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

