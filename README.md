# High-Dimensional-Data-Visualisation
Representing data collected with multiple parameters, with applications to physics.

## Mathematical background

### Projection matrix onto a plane spanned by two vectors

For orthonormal $n$-dimensional vectors $\boldsymbol{v_1}$ and $\boldsymbol{v_2}$, and an $n$-dimensional vector $\boldsymbol{x}$, the projection of $\boldsymbol{x}$ onto the plane spanned by $\boldsymbol{v_1}$ and $\boldsymbol{v_2}$ is given by:
$$
(\boldsymbol{x}\cdot\boldsymbol{v_1}) \boldsymbol{v_1} + (\boldsymbol{x}\cdot\boldsymbol{v_2}) \boldsymbol{v_2}
$$
Now, if we treat $\boldsymbol{v_1}$ and $\boldsymbol{v_2}$ as basis vectors, we can write the projection with two coordinates as
$$
\begin{bmatrix}
    \text{---} & \boldsymbol{v_1} & \text{---}  \\
    \text{---} & \boldsymbol{v_2} & \text{---}
\end{bmatrix}
\boldsymbol{x} = P \boldsymbol{x}
$$
where $P$ is the $2 \times n$ projection matrix onto our plane. 

If we are given two vectors $\boldsymbol{u_1}$ and $\boldsymbol{u_2}$ that are not orthonormal, we can still construct an orthonormalised projection matrix onto the plane they span. First, set $\boldsymbol{v_1} := \boldsymbol{u_1}/|\boldsymbol{u_1}|$ so that $\boldsymbol{v_1}$ is a unit vector. Then, set $\boldsymbol{v_2} := \boldsymbol{u_2} - (\boldsymbol{u_2} \cdot \boldsymbol{v_1}) \boldsymbol{v_1}$ and normalise it to have unit length. Then we can construct $P$ as above.

### Finding the projection of a hyperellipsoid onto a plane

Let $H$ be an $n$-dimensional ellipsoid with equation $\boldsymbol{x}^T A \boldsymbol{x} = 1$, where $A$ is a symmetric, positive definite $n \times n$ matrix. Let $L$ be the plane spanned by orthonormal vectors $\boldsymbol{u}$ and $\boldsymbol{v}$. We aim to project $H$ onto $L$. Let
$$
P:=
\begin{bmatrix}
    \text{---} & \boldsymbol{u} & \text{---}  \\
    \text{---} & \boldsymbol{v} & \text{---}
\end{bmatrix}
$$
be the $2 \times n$ projection matrix onto $L$.

Let $f(\boldsymbol{x}) = \boldsymbol{x}^T A \boldsymbol{x}$. We want to find $\boldsymbol{x}$ for which $\nabla f (\boldsymbol{x})$ is parallel to $L$. These points, when projected onto $L$, will form the boundary of the shadow of $H$.

Since $\nabla f(\boldsymbol{x}) = 2A\boldsymbol{x}$, we must have $A\boldsymbol{x}$ parallel to $L$. That is, there exists some $2 \times 1$ vector $\boldsymbol{s} = (s_1, s_2)$ such that $A \boldsymbol{x} = s_1\boldsymbol{u} + s_2\boldsymbol{v} = P^T \boldsymbol{s}$. We can set $\boldsymbol{x} = A^{-1}P^T\boldsymbol{s}$. For $\boldsymbol{x}$ to be a point on $H$, we require that $1 = \boldsymbol{x}^T A \boldsymbol{x}$; substituting out $\boldsymbol{x}$ gives
$$
    1 = \boldsymbol{x}^T A \boldsymbol{x}\\
    = \boldsymbol{s}^T P (A^{-1})^T A A^{-1} P^T \boldsymbol{s}\\
    = \boldsymbol{s}^T P A^{-1} P^T \boldsymbol{s}
$$
where $(A^{-1})^T = A^{-1}$ is by the symmetry of $A$. We see from this equation that $\boldsymbol{s}$ lies on an ellipse with $2 \times 2$ matrix $P A^{-1} P^T$. 

Now, let $\bar{\boldsymbol{x}} := P \boldsymbol{x}$ be the projection of $\boldsymbol{x}$ onto $L$. For $\bar{\boldsymbol{x}}$ on the boundary of the projected ellipse, we have $\bar{\boldsymbol{x}} = P \boldsymbol{x} = P A^{-1} P^T \boldsymbol{s}$. Then, we can isolate $\boldsymbol{s} = (P A^{-1} P^T)^{-1} \bar{\boldsymbol{x}}$, and, substituting out $\boldsymbol{s}$, find
$$
    1 = \boldsymbol{s}^T P A^{-1} P^T \boldsymbol{s}\\
    = \bar{\boldsymbol{x}}^T ((P A^{-1} P^T)^{-1})^T P A^{-1} P^T (P A^{-1} P^T)^{-1} \bar{\boldsymbol{x}}\\
    = \bar{\boldsymbol{x}}^T (P A^{-1} P^T)^{-1} \bar{\boldsymbol{x}}.
$$
Again, $((P A^{-1} P^T)^{-1})^T = (P A^{-1} P^T)^{-1}$ by symmetry of $A$. 

Therefore, the projection of $H$ onto $L$ is the ellipse with equation $ \bar{\boldsymbol{x}}^T (P A^{-1} P^T)^{-1} \bar{\boldsymbol{x}} = 1$.

### Generating a sequence of points

Having an equation in the form $\bar{\boldsymbol{x}}^T Q^{-1} \bar{\boldsymbol{x}} = 1$ for our projected ellipse, we now need to generate a $2 \times m$ array of points on the boundary of this ellipse. There are many ways to do this, but my approach was to:

* Generate a $2 \times m$ array $C$ of points on the unit circle. This is done only once, on initialisation of the EllipseCalculator, using the parameterisation $(\cos(t), \sin(t))$. Then,
* Construct a $2 \times 2$ transformation matrix $T$ such that $TC$ gives the required $2 \times m$ matrix of points on the ellipse. 

The difficult part is finding $T$, which is done as follows. 

Note that, as $Q$ is positive definite, we can write $Q = BB^T$ for some matrix $B$. Our ellipse equation becomes $\bar{\boldsymbol{x}}^T (B^T)^{-1} B^{-1} \bar{\boldsymbol{x}} = 1$. For each column vector $\boldsymbol{c}$ of $C$, we know $\boldsymbol{c}^T \boldsymbol{c} = 1$. That is, we can write $B^{-1} \bar{\boldsymbol{x}} = \boldsymbol{c}$ for $\bar{\boldsymbol{x}}$ on the boundary of our ellipse. Consequently $\bar{\boldsymbol{x}} = B\boldsymbol{c}$, so that $B$ transforms the circle into the ellipse. Ergo $B$ is $T$, the transformation matrix we are looking for.

We set $Q = TT^T$. Moreover, note that to transform a circle into an ellipse we only need to stretch the ellipse along the $x$ and $y$ axes (applying a diagonal matrix) then rotate the ellipse (applying a rotation matrix). That is, for some $a$, $b$ and $\theta$ yet to be determined,
$$
T =
\begin{bmatrix}
    \cos(\theta) & -\sin(\theta)  \\
    \sin(\theta) & \cos(\theta) 
\end{bmatrix}
\begin{bmatrix}
    a & 0  \\
    0 & b
\end{bmatrix}
$$