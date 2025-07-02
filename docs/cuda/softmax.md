# Softmax: numerical instability
## Softmax

<p align="center">
<img src = "attachments/img-20250701142558.png" width="600">
</p>

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum^K_{j=1}e^{z_j}}
$$

Softmax의 수식은 위와 같다. 이 함수의 의미는 주어진 값들 속에서 각각의 원소의 확률을 계산해준다. 유용한 함수이지만, 실제 컴퓨터상에서 구현하게 될 때는 부동소수점으로 인한 문제가 발생하기 쉽다. 부동소수점은 소수를 컴퓨터로 표현하는 방법인데, 중요한 점은 표현할 수 있는 범위가 한정된다는 것이다. 그런데 softmax 연산에는 지수함수($e^z$)의 특성으로 인해서 값이 아주 커진다. 이 값들의 합연산을 하면 overflow가 발생하기 쉽다.

## Safe softmax
이 문제를 해결하기 위해서 수식을 약간 바꾸게 된다.

$$
\sigma(\mathbf{z})_i = \frac{e^{z_i - \max(z)}}{\sum^K_{j=1}e^{z_j-\max(z)}}
$$

