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
    <!--https://codepen.io/paulobrien/pen/gWoVzN-->
    <search-field-container/>

    <br>

    <!--attempt with side-by-side divs-->
    <div class="mycontainer2">
      <div class="leftdiv2">
        <span class="title">Current Product</span>

      </div>
      <div class="rightdiv2">
        <span class="title">Browse History</span>

      </div>
    </div>

    <!--attempt with side-by-side divs-->
    <div class="mycontainer">
      <div class="leftdiv">
        <!--<span class="title">Current Product</span>-->
        <card :imgUrl="$store.state.currentProduct.productImageUrl"
              :productTitle="$store.state.currentProduct.productTitle"
              productDescription='no description'
              :productId="$store.state.currentProduct.productId"
              :theclick="removeProduct"
              hoverOverlay="❌"
              :stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object="$store.state.currentProduct">
        </card>
      </div>
      <div class="rightdiv">
        <!--<span class="title">Browse History</span>-->

        <card-list :products="$store.state.browsedProducts" :clickToRemove="true"/>
      </div>
    </div>

    <!--attempt with table-->
    <div style="
    overflow-x: scroll;
    overflow-y: hidden;
    padding-bottom: 10px;display:none;">
      <table>
        <tr>
          <th>Current Product</th>
          <th>Browse History</th>
        </tr>
        <tr>
          <td style="border:4px solid blue; padding:10px;">
            <card :imgUrl="$store.state.currentProduct.productImageUrl"
                  :productTitle="$store.state.currentProduct.productTitle"
                  productDescription='no description'
                  :productId="$store.state.currentProduct.productId"
                  :theclick="removeProduct"
                  hoverOverlay="❌"
                  :stupid_extra_variable_to_hold_product_cos_cant_make_anonymous_function_in_card_parameters_to_accept_product_object="$store.state.currentProduct">
            </card>
          </td>
          <td style="border:1px solid black; padding:10px;">
            <card-list :products="$store.state.browsedProducts" :clickToRemove="true"/>
          </td>
        </tr>
      </table>
    </div>

    <hr>

    <span class="title">Search Results</span>
    <!--is loading? {{ $store.state.isLoadingProductsSearch }}  &#45;&#45; num results: {{ $store.state.queriedProducts.length }}-->
    <card-list :products="$store.state.queriedProducts" :isLoading="$store.state.isLoadingProductsSearch.length > 0"/>
    <!--<card :isHidden="!$store.state.isLoadingProductsSearch"-->
    <!--imgUrl="https://i.allthepics.net/2017/08/06/fidgit-spinner-dragonb1f6d.gif"-->
    <!--productTitle="Loading..."-->
    <!--productId="...">-->
    <!--</card>-->

    <hr>
    <span class="title">Bronto Recs</span> &nbsp;&nbsp; (estimated by search Bronto's "viewed this / viewed that" indicator lists in Solr)
    <!--is loading? {{ $store.state.isLoadingBrontoRecommendedProducts }}-->
    <br/>
    <card-list :products="$store.state.brontoRecommendedProducts" :isLoading="$store.state.isLoadingBrontoRecommendedProducts.length > 0"/>
    <!--<card :isHidden="!$store.state.isLoadingBrontoRecommendedProducts"-->
    <!--imgUrl="https://i.allthepics.net/2017/08/06/fidgit-spinner-dragonb1f6d.gif"-->
    <!--productTitle="Loading..."-->
    <!--productId="...">-->
    <!--</card>-->

    <hr>
    <span class="title">Neural Network Recs</span>
    <!--is loading? {{ $store.state.isLoadingStuartRecommendedProducts }}-->
    <br/>
    <card-list :products="$store.state.stuartRecommendedProducts" :isLoading="$store.state.isLoadingStuartRecommendedProducts.length > 0"/>
    <!--<card :isHidden="!$store.state.isLoadingStuartRecommendedProducts"-->
    <!--imgUrl="https://i.allthepics.net/2017/08/06/fidgit-spinner-dragonb1f6d.gif"-->
    <!--productTitle="Loading..."-->
    <!--productId="...">-->
    <!--</card>-->

  </div>
</template>

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
        'browse',
        'removeProduct'
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
<style>
  hr {
    border-top: 1px solid #8c8b8b;
  }

  th, span.title {
    font-size: 30px;
    font-weight: bold;
  }

  th {
    padding-left: 10px;
    padding-right: 10px;
  }

  .mycontainer {
    text-align: left;
    width: 100vw;
    /*display: inline;*/

    /*position:relative*/
    /*padding: 15px;*/
  }

  .leftdiv {
    display: inline-block;
    width: 190px;
    /*max-width: 300px;*/
    text-align: left;
    /*bottom: 0;*/
    /*padding: 30px;*/
    /*background-color: #ddd;*/
    /*border-radius: 3px;*/
    /*margin: 15px;*/
    /*vertical-align: top;*/
    padding-top: 15px;

    position: relative;
    min-height:200px;
    border: 4px solid blue;

  }

  .rightdiv {
    display: inline;
    position: absolute;
    /*float: right;*/
    left: 215px;
    right: 10px;

    /*bottom: 0;*/
    vertical-align: middle;
    /*top: 50%;*/

    /*max-width: 80%;*/
    text-align: left;
    /*padding: 30px;*/
    /*background-color: #ddd;*/
    /*border-radius: 3px;*/
    padding-top: 15px;
    padding-left: 1px;
    min-height:180px;
    border: 1px solid black;

  }


  .mycontainer2 {
    text-align: left;
    width: 100%;
    /*display: inline;*/

    position:relative
    /*padding: 15px;*/
  }

  .leftdiv2 {
    display: inline-block;
    width: 190px;
    /*max-width: 300px;*/
    text-align: center;
    bottom: 0;
    /*padding: 30px;*/
    /*background-color: #ddd;*/
    /*border-radius: 3px;*/
    /*margin: 15px;*/
    vertical-align: top;
    position: relative;
    padding-left:6px;
    margin-top:-20px;

  }

  .rightdiv2 {

    display: inline;
    position: absolute;
    /*float: right;*/
    left: 205px;
    right: 10px;
    top: 30%;
    /*right:50%;*/
    /*top: 40px;*/
    /*bottom: 0;*/
    vertical-align: middle;
    /*top: 50%;*/

    /*max-width: 80%;*/
    text-align: center;
    /*padding: 30px;*/
    /*background-color: #ddd;*/
    /*border-radius: 3px;*/
    /*margin: 15px;*/
  }
</style>
