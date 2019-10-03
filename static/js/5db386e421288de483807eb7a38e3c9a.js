(function (root, factory) {if (typeof define === 'function' && define.amd) {define(['exports', 'echarts'], factory);} else if (typeof exports === 'object' && typeof exports.nodeName !== 'string') {factory(exports, require('echarts'));} else {factory({}, root.echarts);}}(this, function (exports, echarts) {var log = function (msg) {if (typeof console !== 'undefined') {console && console.error && console.error(msg);}};if (!echarts) {log('ECharts is not Loaded');return;}if (!echarts.registerMap) {log('ECharts Map is not loaded');return;}echarts.registerMap('柘荣县', {"type":"FeatureCollection","features":[{"type":"Feature","id":"350926","properties":{"name":"柘荣县","cp":[119.900609,27.233933],"childNum":1},"geometry":{"type":"Polygon","coordinates":["@@A@A@@@AAAA@@B@@A@@@@AA@@A@C@CAA@A@AACAA@EACAA@@C@@@AA@@@A@@B@BA@ACCAAAA@A@@@AAA@@@@@A@AA@@@@@@@@@@A@@@@@A@@@@@@@@@A@@@B@@B@@@@@B@@B@@B@@@B@@@BB@@@@B@@@@BB@@A@@@@@@@@@@@@@A@@B@@@@@@A@@@@@@@AB@@@@A@@@@@@@@A@@@@@A@@@A@@@@@@@@@@@@@@@@A@@@@@AA@@@@@@@@A@@@AA@@@B@@@@@@@@@@B@@BB@@@@@BB@@A@BB@B@@@@@@BB@@B@BA@@@@@@B@@@BA@@B@@BB@@BBB@@@@@@B@@@BD@@@@@B@@A@AB@@@@AB@@B@@@@B@@@@BBB@@@B@@@@@@@@@ABA@@@@B@@@B@@BB@@B@BBB@BB@@@@@@@@BB@@@@@@@@A@ECAAA@A@A@C@A@C@C@A@A@AB@@ABA@@BA@MAAAA@A@@@A@E@A@A@@@AA@@@@@A@@@A@A@@@A@A@A@GAC@@CAAAA@C@C@E@A@A@C@ABC@AB@@CAA@A@ABC@AB@@@@CAA@IB@@A@ABAB@@AB@@B@@@BAB@B@@@@B@BAB@DA@@B@@C@AB@BA@BD@@D@BB@@@@BBBB@@@@ABA@BB@@@@BBB@BB@@@@@BA@@@@@@BA@@@@@AB@@@@ABAC@@A@@@A@ABA@A@A@CBC@EBAB@AA@@@@@A@A@@@A@AACBA@AB@BA@@@@F@@@@@B@D@B@B@BABB@BD@@@@@@A@@@A@@BA@B@@B@@A@@@A@@AA@A@@BA@A@ADA@@BABADAB@@C@@@ADCF@BA@BB@@BB@BBB@B@@BB@B@B@@@BAB@B@@BBBBB@@@@BD@B@BA@@BAB@B@@B@@BBBD@@@B@DA@@@@D@@@D@B@B@@D@@B@@BBBB@@@BA@@BAB@@@@AAA@@B@D@@@@@@@@CAA@AA@@A@@BA@@@@B@@@@AA@@@@A@@B@AAB@@@@AB@@AB@@@@@BB@@@@@AB@@@@@@@BB@@B@@@@@BA@@@@@@@@BA@@B@@@@@BA@@@@@BBB@@@B@ABB@@@@@@BB@@@@B@@@BB@@@@B@@@BBBD@BBB@BB@@B@@@@@@AB@@AB@@@@ABA@@BA@@@@BBBB@@BDB@@DB@BBBDB@@B@@AB@@@@BB@@@B@@@@@@B@@@BB@@ABB@@@@@@BBF@@BB@B@BBD@@BB@@@B@@@BB@BBB@B@B@D@B@B@DA@@B@BBBBDBBBBBFBB@BB@@@@@@AB@@AB@@AA@@A@@@A@@@@@@BB@B@EDABAB@@@@@BH@@@@B@B@@@D@@@@@BB@@B@@@B@@@BABABA@A@A@ABA@ABCBCBCDEBABABCDCBGF@@@@@BA@@BA@@@@BB@DDBB@@B@BB@@@@BA@@@@@@@A@@B@@AB@B@B@B@@@B@B@B@DB@@@@HE@AB@BAB@@BB@B@@BBB@BA@A@@B@@CBCB@@@@BBBB@@B@B@D@B@@@BB@@@@AB@@CBA@C@@BA@@@@B@@BB@BBB@B@@@@@BBA@BDD@@@BB@@DB@@B@@BBDFBBBDBBBB@B@B@@@BGB@BA@A@ABA@ABC@A@A@@B@@@BABD@@@B@@B@@@@CB@BA@@BB@B@@@D@@@DAB@BAB@DAB@D@D@B@@@DA@@B@BB@B@DBB@B@B@BB@@@@BB@BAD@BAD@BAB@FCB@BAB@B@BA@@BA@A@@BCBABAB@BADAB@B@B@@@BB@@@B@D@BB@@B@@@@B@@@B@@@@A@@B@@@B@@@@@@@@@@@@@@A@@@@BAB@DBB@@@BBB@B@B@B@DABAB@B@B@@BB@BD@BBD@B@BA@A@C@ABA@CBAD@@ABA@@BB@BB@@BDDDBDDDBBBBB@B@B@BABA@AB@@@@AB@BABABA@@BA@@BA@@@@@@B@@@BBBB@@B@B@BAB@B@BA@@BB@@BAB@B@B@BBBA@@@@@@B@@AA@@A@@AA@AA@A@AAA@A@AA@@@A@A@@@AB@B@@@D@B@BBD@@@B@@@@AB@AC@A@A@@BA@@BAB@@@BB@@BB@B@F@DBD@@@@B@B@BA@@B@@@@@@AB@@AB@@@@B@@B@@B@B@@@B@B@BB@@@@@B@BAB@BABA@@B@@@BBB@@@B@B@B@B@@@@@@B@B@DB@@@@BBBB@BBD@B@@@@@B@@@B@@@B@B@@@B@B@D@@@BABABABAB@@AB@@@@@@@DB@B@@AB@@B@@@BBBAB@F@@@B@B@B@B@B@@@BB@CAA@AA@AACCAA@@@@@A@@BABB@@B@DCB@@@B@B@B@@@B@@@@A@A@A@@@@B@B@B@@@B@@A@@@@@CA@@AA@@@@@@CAE@A@@@@BA@A@@B@B@B@F@BAB@BAFABB@B@B@@B@B@@ABA@@B@B@F@D@B@B@@@@@@A@@CAA@@A@@@A@@@AAAA@@AB@BA@ADB@AB@BADADA@@@A@A@@@@AAA@@A@@@A@A@A@A@AAC@@BA@A@@@A@@@@B@@@@A@AB@@@@ABA@CDCDCBAB@FCBAB@BAB@B@B@DA@@BA@@D@@@@A@@A@AAAA@@@A@@@@BAFAD@DAD@DA@@@A@A@A@@@A@@@AB@B@B@DCD@DADA@A@@@A@@@A@ABCDA@AD@@@@A@A@C@C@A@@@@B@DA@@BBBBFBDBD@@@B@BABCB@@AD@FAB@BADCBAB@BAD@B@BBB@B@BAFAD@@@@A@@D@@AB@@@@A@E@C@@BC@@@A@ABA@ABA@@BEBC@A@@@C@@@@@AJ@F@BAB@BA@@@A@C@@@A@@BA@@AAA@A@AA@@A@@C@@A@EAA@AA@ACA@AA@@ABA@C@@BAB@@@@@@A@C@@@@AA@C@@@A@@AAA@A@@@A@AB@@CB@A@@A@BA@ABA@AAA@@@AA@A@AA@B@BAB@@@B@@@B@@AB@@@B@@A@@B@@A@@@@@A@@B@@@B@B@@AB@BAB@@A@@BC@GAC@C@A@A@@@CA@@ABA@A@A@E@CBCAC@A@C@E@CBC@A@@@@@@D@BA@@@GBA@@@CD@B@@A@AB@@AB@@@B@@A@A@@@AAA@ABCB@@A@@A@@@@@A@@@@AA@@@A@@CC@A@A@@AA@@@AA@@@BAB@@@@@@@B@@@@@@@@@BA@@@@@@@A@@B@@@@AB@@BDA@@@@@@@A@@@@@A@@@@@@EBMHIFAEFEFCBIAG@CDEFMBG@CBCBKEII@GBA@ACA@@@AA@@ABAA@@@@@A@@@@A@A@A@@@@A@A@@CA@@A@@@AA@@@AAC@C@C@CAAAC@AA@AAAA@AAC@ACEAA@@@A@BA@@@A@EAA@@@A@ABA@ABA@@@AB@@@@A@A@ABAAA@@@@@A@@@A@@@@@CD@BA@ABAB@BA@ABA@@BB@BF@@@BBB@B@@AB@@@@@B@@ABA@A@@@@@@@AA@@A@A@@@@@AB@@@@EAC@A@@AA@@@@@A@AAABA@@AA@@@A@@@@AA@@@A@C@AC@@@A@A@@@@A@A@A@@@A@@A@@CAAA@@ABAAA@@@A@A@@@A@A@@@A@A@A@@@@B@B@@CB@B@@ABABA@@@C@@B@@@@@@ABAAAACAA@@ACBAAA@AAA@AC@@A@"],"encodeOffsets":[[122745,27752]]}}],"UTF8Encoding":true});}));