document
.getElementById("loanForm")
.addEventListener(
"submit",
async function(e){

e.preventDefault();

const resultCard =
document.getElementById(
"resultCard"
);

const resultDiv =
document.getElementById(
"result"
);

resultCard.className =
"card p-4";

resultDiv.innerHTML =
"⏳ Predicting...";

const payload = {

Gender:
document.getElementById(
"Gender"
).value,

Married:
document.getElementById(
"Married"
).value,

Dependents:
document.getElementById(
"Dependents"
).value,

Education:
document.getElementById(
"Education"
).value,

Self_Employed:
document.getElementById(
"Self_Employed"
).value,

ApplicantIncome:
parseFloat(
document.getElementById(
"ApplicantIncome"
).value
),

CoapplicantIncome:
parseFloat(
document.getElementById(
"CoapplicantIncome"
).value
),

LoanAmount:
parseFloat(
document.getElementById(
"LoanAmount"
).value
),

Loan_Amount_Term:
parseFloat(
document.getElementById(
"Loan_Amount_Term"
).value
),

Credit_History:
parseFloat(
document.getElementById(
"Credit_History"
).value
),

Property_Area:
document.getElementById(
"Property_Area"
).value

};

try{

const response =
await fetch(
"/predict",
{
method:"POST",

headers:{
"Content-Type":
"application/json"
},

body:
JSON.stringify(
payload
)
}
);

const result =
await response.json();

if(
result.result ===
"Approved"
){

resultCard.className =
"card p-4 approved";

resultDiv.innerHTML =

`
<div>

✅ LOAN APPROVED

<br><br>

Confidence

<br>

${result.confidence}%

</div>
`;

}
else{

resultCard.className =
"card p-4 rejected";

resultDiv.innerHTML =

`
<div>

❌ LOAN REJECTED

<br><br>

Confidence

<br>

${result.confidence}%

</div>
`;

}

}
catch(error){

console.error(error);

resultDiv.innerHTML =

`
❌ ERROR
`;

}

}
);