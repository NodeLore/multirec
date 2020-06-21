'use strict'

let padding = 30
let lineLimit = 20
let model = "ArangoDB"

function chineseCount(str){
    var trim= (str||"").replace(/^(\s|\u00A0)+|(\s|\u00A0)+$/g,"");//+表示匹配一次或多次，|表示或者，\s和\u00A0匹配空白字符，/^以……开头，$以……结尾，/g全局匹配,/i忽略大小写
    var strlength=str.length;
    if(!strlength){   //如果字符串长度为零，返回零
        return 0;
    }
    var chinese=str.match(/[\u4e00-\u9fa5]/g); //匹配中文，match返回包含中文的数组
    return chinese==null?0: chinese.length; //计算字符个数
}

function setSize(tArea){
    let text = tArea.text().trim()
    let cCount = chineseCount(text)
    let width = tArea.width()
    let height = tArea.height()
    let limit = ((width-padding*2) / text.length) < lineLimit ? lineLimit : (width-padding*2) / text.length
    
    let modifyParams = {
        1: 0.2,
        2: 0.5,
        3: 0.8,
        4: 0.9
    }
    let modify = modifyParams[text.length] == null ? 1 : modifyParams[text.length]
    let size = modify * limit * (1 + 0.5 * ((text.length-cCount)/text.length))
    tArea.animate({'font-size': size + 'px', 'line-height': size*text.length > (width-2*padding)*2.7? size : height + 'px'})
}

function recommendMovie(type, key){
    $('#loading').show()
    if(type == 'storyline'){
        $('#modelName').text(model)
        $('#recKey').text($('#titleArea').text().split('/')[0].trim())
    }
    else{
        $('#modelName').text('ArangoDB')
        $('#recKey').text(key)
    }

    $.ajax({
        type: 'GET',
        url: '/recommend/?model=' + model + '&type=' + type + '&key=' + key,
        success: function(data){
            if(data['result'] && data['result'] != []){
                $('.resultItem').remove()
                console.log(data['result'])
                for(var i = 0; i < data['result'].length; i++){
                    let item = data['result'][i]
                    console.log(item)
                    let divItem = $('<div></div>')
                    divItem.addClass('resultItem')
                    divItem.text(item.name.split('/')[0])
                    divItem.attr('value', item.key)
                    
                    $(divItem).click(function(){
                        $('#loading').show()
                        location.href = '/movie/?key=' + item.key
                    })

                    $('#recommendResult').append(divItem)
                }
                $('#mask').show()
                $('#recommendResult').fadeIn()
            }
            $('#loading').hide()
        }
    })
}

function updatePerson(key){
    $.ajax({
        type: 'GET',
        url: '/person/?key=' + key,
        success: function(data){
            if(data['result'] && data['result'] != {}){
                $('#infoName').text(data['result'].name)
                $('#infoRole').text(data['result'].prof)
                $('#infoName').attr('value', data['result'].key)
                let works = data['result'].works
                $('#information ul').children().remove()
                for(var i = 0; i < works.length; i++){
                    let work = works[i]
                    let liItem = $('<li></li>')
                    liItem.text(work.name)
                    liItem.attr('value', work.key)
                    $('#information ul').append(liItem)

                    $(liItem).click(function(){
                        $('#loading').show()
                        location.href = '/movie/?key=' + work.key
                    })
                }
                $('#mask').show()
                $('#information').fadeIn()
            }
        }
    })
}

function updateByKeywords(word){
    $('#loading').show()
    $.ajax({
        type: 'GET',
        url: '/search/?keyword=' + word,
        success: function(data){
            if(data['result'] && data['result'] != []){
                $('#searchKey b').text(word)
                $('#resultPanel ul').children().remove()
                for(var i = 0; i < data['result'].length; i++){
                    let item = data['result'][i]
                    let liItem = $('<li></li>')
                    liItem.text(item.name)
                    liItem.attr('value', item.key)
                    $('#resultPanel ul').append(liItem)
                    
                    $(liItem).click(function(){
                        $('#loading').show()
                        location.href = '/movie/?key=' + item.key
                    })
                }
                $('#mask').show()
                $('#resultPanel').fadeIn()
            }
            $('#loading').hide()
        }
    })
}

$(document).ready(function(){
    window.onload = function(){
        Grade(document.querySelectorAll('.gradient-wrap'))
    }

    let titleArea = $('#titleArea')
    setTimeout(()=>{
        setSize(titleArea)
    }, 500)
    
    $(this).keyup(function(e){
        if(e.keyCode == 27){
            $('.modalWin').slideUp()
            $('#mask').hide()
        }
        else if(e.keyCode == 13 && $('#searchInput').is(':focus') && $('#searchInput').val() != ''){
            if($('#resultPanel').is(':hidden')){
                let keyword = $('#searchInput').val()
                updateByKeywords(keyword)
            }
        }
    })

    let currentModel = $('#modelSelect div:eq(0)')
    $('#modelSelect div').click(function(){
        if($(this) != currentModel){
            currentModel.removeClass('selected')
            currentModel = $(this)
            currentModel.addClass('selected')
            model = $(this).attr('value')
        }
    })

    $('#mask').click(function(){
        $('.modalWin').slideUp()
        $('#mask').hide()
    })

    let originText = '';
    $('.resultItem').mouseenter(function(){
        originText = $(this).html()
        $(this).html($(this).attr('value'))
    })

    $('.resultItem').mouseleave(function(){
        $(this).html(originText)
    })

    $('#recommendBtn').click(function(){
        var key = $('#titleArea').attr('value')
        if(key != ''){
            recommendMovie('storyline', key)
        }
    })

    $('#directorArea ul li').click(function(){
        let value = $(this).attr('value')
        if(value != '')
        {
            updatePerson(value)
        }
    })

    $('#castArea ul li').click(function(){
        let value = $(this).attr('value')
        console.log(value)
        if(value != '')
        {
            updatePerson(value)
        }
    })

    $('#personRec').click(function(){
        let key = $('#infoName').attr('value')
        if(key && key != ''){
            recommendMovie('person', key)
        }
        $('#mask').hide()
        $('#information').fadeOut()
    })

    $('.movieTag').click(function(){
        let key = $(this).text()
        if(key != ''){
            recommendMovie('tag', key)
        }
    })
})