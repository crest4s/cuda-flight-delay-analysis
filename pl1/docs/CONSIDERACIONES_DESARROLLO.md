# Consideraciones de Desarrollo

## 2.1. Consideraciones Iniciales

### Ejecución en GPU
- **Obligatorio**: Salvo ciertas acciones (por ejemplo, leer/guardar datos de/en disco, gestión del menú de opciones), que puede ser realizado desde la parte de la CPU del programa, las diferentes partes solicitadas **obligatoriamente deberán realizarse en la GPU** mediante programación en CUDA C.

### Excepción para CUDA Virtualizado
- **Casos especiales**: Si se hace uso de CUDA virtualizado (para gráficas muy antiguas), se permite realizar alguna acción desde el host. Esto es debido a que la versión de CUDA de la máquina virtual no tiene implementación de algunas funciones de CUDA. 
  - ⚠️ **Importante**: Consultar con los profesores antes de proceder con esta excepción.

### Salidas del Programa
- Las salidas principales del programa serán **vía consola**.
- **No se sobrescribirá el dataset original**.
- La salida vía consola deberá incluir **trazas para seguir la ejecución del programa**.

### Acceso a Memoria
- El acceso a los datos en memoria global de la GPU será **obligatoriamente realizado mediante acceso linealizado (1D)**.

### Implementación de Código
- Las modificaciones sobre los datos se realizarán mediante **código C propio**, escrito en kernels de CUDA.
- **No se permite la utilización de librerías externas**, como OpenCV o similares.
- **No se permite** el uso de algunas funciones que implementa CUDA con codificaciones avanzadas para patrones de código.

### Documentación del Código
- **Obligatorio**: Todo el código entregado deberá estar **perfectamente comentado**.
- ⚠️ **Advertencia**: Aquella práctica que no tenga el código bien documentado puede ser suspendida.

### Recomendaciones para Desarrollo
- Se recomienda **comenzar la codificación con un subconjunto del dataset original** (decenas de datos).
- Usar unas **dimensiones iniciales que permitan ejecutar el programa de forma rápida**.
- Iniciar con **un único bloque** (Stream multiprocesador, SM) y el uso de la memoria global.
- Esto facilita la codificación inicial.
- **Nota**: La cantidad máxima de hilos por bloque es dependiente del hardware, pero en general sirve para los primeros pasos.

---

**Fecha de actualización**: 26 de febrero de 2026
