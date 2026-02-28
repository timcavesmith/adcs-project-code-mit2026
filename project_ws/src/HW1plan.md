# ADCS HW 1: 

# Part 4: 3-DOF 3D 2-body orbital dynamics sim 

- ideally later we can expand it to be 6-DOF 3D with attitude dynamics

## packages:
- use matplotlib for basic output plots
- perhaps try pyvista for attitude dynamics modeling later (whatever david used)


## units:
- km
- seconds

frame: ECI (earth-centered inertial)

## plan:

init params:
R, V
celestial body mass and radius



def dynamics()
    our DE

n_steps = sim_time / dt 

def rk4(xdot, x, x0, dt, n_steps)
    for i in range(n_steps)
        the rk4 steps


plotting and visualization:
- plotly for earth sim

## Relevant eqns:

From Newton's law of gravitation and 3rd law we have:

$ F = ma$
and 

$F = \frac{GMm}{r^2}$ 

so 2-body EOM is

$-\frac{GM}{r^3}\bar x = \bar a$

so for,
$$

x = \begin{bmatrix} \bar r \\ \bar v\end{bmatrix} \in \mathbb{R}^6\\[10pt]
\dot x = \frac{dx}{dt} = \begin{bmatrix} v \\ a \end{bmatrix}$$

and GM is universal gravitational constant times Earth's mass which is earth's gravitational parameter, $\mu$


# Part 5: 3-DOF 3D attitude dynamics sim

The states are,
$$

x = \begin{bmatrix} q \\ \omega \end{bmatrix} \in \mathbb{R}^{10}\\[10pt]

\dot x = \frac{dx}{dt} = \begin{bmatrix} \dot q \\ \dot \omega \end{bmatrix}.

$$

The EOMs are quaternion kinematics and Euler's eqns:

$$

\dot q = \frac{1}{2}L(q)H\omega = \frac{1}{2}G(q)\omega 

\\[10pt]
\dot\omega =  J^{-1}(\tau - \omega \times J\omega)
$$

writen in body (or better, in principal axes). Can use np.linalg.solve but a 3x3 inverse is trivial. Also, if using principal axis J, can just do elementwise divide since we'd have diagonal J.


    




















# for future reference for making md files
NOTE: just hitting Enter once doesn't create a new line.  

Two spaces at the end of a line then Enter — creates a line break
Two Enters (blank line) — creates a new paragraph (slightly more spacing)
**bold**
*italic*
~~strikethrough~~

- bullet
- bullet
  - nested bullet

1. numbered
2. list

> blockquote

`inline code`

​```python
code block
​```

[link text](https://example.com)

![image alt text](image.png)

---
(horizontal rule)

| Column 1 | Column 2 | Column 3 |
|-----------|----------|----------|
| data      | data     | data     |
| data      | data     | data     |

(recommened Markdown All in One for table formatting)

- [ ] unchecked task
- [x] completed task